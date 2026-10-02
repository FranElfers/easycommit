// Package server manages the background llama-server instance that easycommit
// owns: it starts it, finds it again on later runs and stops it.
package server

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/franelfers/easycommit/internal/config"
	"github.com/franelfers/easycommit/internal/llama"
)

const (
	pollInterval = 200 * time.Millisecond
	stopTimeout  = 10 * time.Second
	// maxSocketPath is the longest path a Unix socket address can hold on Linux.
	maxSocketPath = 107
)

// State describes the server as seen by `easycommit server status`.
type State string

const (
	Stopped  State = "not running"
	Starting State = "starting"
	Loaded   State = "running, model loaded"
	Unloaded State = "running, model unloaded (idle)"
)

// Server is the instance whose state files live in Dir.
type Server struct {
	Dir string
}

// New returns the server of the current user, creating its state directory.
func New() (*Server, error) {
	s := &Server{Dir: filepath.Join(os.Getenv("XDG_RUNTIME_DIR"), "easycommit")}
	if os.Getenv("XDG_RUNTIME_DIR") == "" {
		// No session manager set it: fall back to a private per-user directory.
		cache, err := os.UserCacheDir()
		if err != nil {
			return nil, err
		}
		s.Dir = filepath.Join(cache, "easycommit", "run")
	}
	if len(s.socket()) > maxSocketPath {
		return nil, fmt.Errorf("state directory path is too long for a Unix socket: %s", s.Dir)
	}
	return s, os.MkdirAll(s.Dir, 0o700)
}

func (s *Server) socket() string  { return filepath.Join(s.Dir, "server.sock") }
func (s *Server) pidFile() string { return filepath.Join(s.Dir, "server.pid") }
func (s *Server) logFile() string { return filepath.Join(s.Dir, "server.log") }

// FindBinary resolves the llama-server executable or explains how to install it.
func FindBinary(name string) (string, error) {
	bin, err := exec.LookPath(name)
	if err != nil {
		return "", fmt.Errorf(`%s not found (%w). Install llama.cpp with Vulkan support:
  Arch Linux:   pacman -S llama-cpp ggml-vulkan
  Alpine Linux: apk add llama-server llama.cpp-vulkan
or point server_bin (--server-bin) at an existing llama-server`, name, err)
	}
	return bin, nil
}

// Ensure returns a client for a ready server running with cfg, starting or
// restarting the instance when needed. Progress goes to stderr.
func (s *Server) Ensure(ctx context.Context, bin string, cfg config.Config, stderr io.Writer) (*llama.Client, error) {
	unlock, err := s.lock(stderr)
	if err != nil {
		return nil, err
	}
	defer unlock()

	// The command line that starts the server also identifies a running one.
	want := s.args(cfg, modelArgs(cfg.Model))

	pid, running := s.process()
	if pid != 0 && !hasSuffix(running, want) {
		fmt.Fprintln(stderr, "easycommit: settings changed, restarting llama-server")
		s.terminate(pid)
		pid = 0
	}

	follow := false
	if pid == 0 {
		fmt.Fprintf(stderr, "easycommit: starting llama-server (context %d tokens, %s)\n", cfg.CtxSize, offload(cfg))
		// A repository may have to be downloaded, which only the server log shows.
		follow = slices.Contains(want, repoFlag)
		if pid, err = s.start(bin, want); err != nil {
			return nil, err
		}
	}
	client := llama.NewClient(s.socket())
	return client, s.waitReady(ctx, client, pid, follow, stderr)
}

// Status reports the state of the server and its PID when it is running.
func (s *Server) Status(ctx context.Context) (State, int) {
	pid, _ := s.process()
	if pid == 0 {
		return Stopped, 0
	}
	sleeping, err := llama.NewClient(s.socket()).Sleeping(ctx)
	switch {
	case err != nil:
		return Starting, pid
	case sleeping:
		return Unloaded, pid
	}
	return Loaded, pid
}

// Stop terminates the server and reports whether there was one running.
func (s *Server) Stop(stderr io.Writer) (bool, error) {
	unlock, err := s.lock(stderr)
	if err != nil {
		return false, err
	}
	defer unlock()
	pid, _ := s.process()
	if pid == 0 {
		return false, nil
	}
	s.terminate(pid)
	return true, nil
}

// args is the full llama-server command line for cfg. It doubles as the
// identity of a running instance: see process.
func (s *Server) args(cfg config.Config, model []string) []string {
	args := []string{
		"--host", s.socket(),
		"--ctx-size", strconv.Itoa(cfg.CtxSize),
		"--gpu-layers", strconv.Itoa(cfg.GPULayers),
		"--sleep-idle-seconds", strconv.Itoa(cfg.IdleTimeout),
		// One slot gets the whole context; easycommit never sends parallel requests.
		"--parallel", "1",
	}
	if !cfg.GPU() {
		args = append(args, "--device", "none")
	}
	return append(args, model...)
}

func offload(cfg config.Config) string {
	switch {
	case !cfg.GPU():
		return "CPU only"
	case cfg.GPULayers >= config.AllLayers:
		return "all layers on GPU"
	}
	return fmt.Sprintf("%d layers on GPU", cfg.GPULayers)
}

// lock serializes start and stop across concurrent easycommit runs.
func (s *Server) lock(stderr io.Writer) (unlock func(), err error) {
	f, err := os.OpenFile(filepath.Join(s.Dir, "server.lock"), os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, err
	}
	fd := int(f.Fd())
	if err = syscall.Flock(fd, syscall.LOCK_EX|syscall.LOCK_NB); errors.Is(err, syscall.EWOULDBLOCK) {
		fmt.Fprintln(stderr, "easycommit: waiting for another easycommit run to finish starting the server")
		err = syscall.Flock(fd, syscall.LOCK_EX)
	}
	if err != nil {
		f.Close()
		return nil, err
	}
	return func() { f.Close() }, nil
}

// process returns the PID and command line of the running server, or zero.
// The command line must mention our socket, which rules out a recycled PID
// and any llama-server that easycommit did not start.
func (s *Server) process() (int, []string) {
	data, err := os.ReadFile(s.pidFile())
	if err != nil {
		return 0, nil
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(data)))
	if err != nil {
		return 0, nil
	}
	raw, err := os.ReadFile(fmt.Sprintf("/proc/%d/cmdline", pid))
	if err != nil {
		return 0, nil
	}
	args := strings.Split(strings.TrimSuffix(string(raw), "\x00"), "\x00")
	if !slices.Contains(args, s.socket()) {
		return 0, nil
	}
	return pid, args
}

// start launches the server detached from this process and records its PID.
func (s *Server) start(bin string, args []string) (int, error) {
	// A server that was killed leaves its socket behind.
	os.Remove(s.socket())
	log, err := os.Create(s.logFile())
	if err != nil {
		return 0, err
	}
	defer log.Close()

	cmd := exec.Command(bin, args...)
	cmd.Stdout, cmd.Stderr = log, log
	// Its own session keeps the server alive after easycommit and its terminal exit.
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	if err := cmd.Start(); err != nil {
		return 0, err
	}
	pid := cmd.Process.Pid
	go cmd.Wait() // reap the child if it dies while we are still running
	return pid, os.WriteFile(s.pidFile(), []byte(strconv.Itoa(pid)+"\n"), 0o600)
}

// waitReady blocks until the server answers health checks. With follow set
// it relays the server log, which carries the progress of a model download.
func (s *Server) waitReady(ctx context.Context, client *llama.Client, pid int, follow bool, stderr io.Writer) error {
	var shown int64
	relay := func() {
		if !follow {
			return
		}
		if f, err := os.Open(s.logFile()); err == nil {
			defer f.Close()
			f.Seek(shown, io.SeekStart)
			n, _ := io.Copy(stderr, f)
			shown += n
		}
	}
	for {
		err := client.Health(ctx)
		relay()
		if err == nil {
			return nil
		}
		if !alive(pid) {
			return fmt.Errorf("llama-server exited during startup:\n%s", tail(s.logFile(), 15))
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(pollInterval):
		}
	}
}

// terminate stops the server process group and removes its state files.
func (s *Server) terminate(pid int) {
	signal := func(sig syscall.Signal) {
		// The server leads its own process group, which also covers wrapper scripts.
		if syscall.Kill(-pid, sig) != nil {
			syscall.Kill(pid, sig)
		}
	}
	signal(syscall.SIGTERM)
	deadline := time.Now().Add(stopTimeout)
	for alive(pid) {
		if time.Now().After(deadline) {
			signal(syscall.SIGKILL)
			deadline = time.Now().Add(stopTimeout)
		}
		time.Sleep(pollInterval / 4)
	}
	os.Remove(s.pidFile())
	os.Remove(s.socket())
}

// alive reports whether pid is a live process, not a zombie.
func alive(pid int) bool {
	raw, err := os.ReadFile(fmt.Sprintf("/proc/%d/cmdline", pid))
	return err == nil && len(raw) > 0
}

func hasSuffix(args, suffix []string) bool {
	return len(args) >= len(suffix) && slices.Equal(args[len(args)-len(suffix):], suffix)
}

// tail returns the last n lines of the file at path.
func tail(path string, n int) string {
	data, _ := os.ReadFile(path)
	lines := bytes.Split(bytes.TrimRight(data, "\n"), []byte("\n"))
	return string(bytes.Join(lines[max(0, len(lines)-n):], []byte("\n")))
}
