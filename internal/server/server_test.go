package server

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/franelfers/easycommit/internal/config"
	"github.com/franelfers/easycommit/internal/llamatest"
)

func TestMain(m *testing.M) {
	llamatest.Main()
	os.Exit(m.Run())
}

// setup isolates the state directory and returns the
// server, the fake llama-server binary and a configuration to start it with.
func setup(t *testing.T) (*Server, string, config.Config) {
	t.Helper()
	t.Setenv("XDG_RUNTIME_DIR", llamatest.ShortDir(t))
	s, err := New()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Stop(os.Stderr) })
	cfg := config.Default()
	cfg.VRAMMB = 16304
	return s, llamatest.Install(t), cfg.WithProfile()
}

func TestLifecycle(t *testing.T) {
	ctx := context.Background()
	s, bin, cfg := setup(t)
	var stderr bytes.Buffer

	if state, _ := s.Status(ctx); state != Stopped {
		t.Fatalf("before start: %s", state)
	}
	client, err := s.Ensure(ctx, bin, cfg, &stderr)
	if err != nil {
		t.Fatal(err)
	}
	if n, err := client.CountTokens(ctx, "two words"); err != nil || n != 2 {
		t.Errorf("client is not usable: %d, %v", n, err)
	}
	state, pid := s.Status(ctx)
	if state != Loaded || pid == 0 {
		t.Fatalf("after start: %s, pid %d", state, pid)
	}
	// The model is a repository, which the server may have to download: its
	// output is relayed.
	if !strings.Contains(stderr.String(), "fake download progress") {
		t.Errorf("download progress was not relayed: %q", stderr.String())
	}

	// Same settings: the running instance is reused.
	stderr.Reset()
	if _, err := s.Ensure(ctx, bin, cfg, &stderr); err != nil {
		t.Fatal(err)
	}
	if _, again := s.Status(ctx); again != pid || stderr.Len() != 0 {
		t.Errorf("server was not reused: pid %d -> %d, stderr %q", pid, again, stderr.String())
	}

	// Different settings: the instance is replaced.
	cfg.CtxSize = 8192
	if _, err := s.Ensure(ctx, bin, cfg, &stderr); err != nil {
		t.Fatal(err)
	}
	_, replaced := s.Status(ctx)
	if replaced == pid || alive(pid) {
		t.Errorf("server was not restarted: pid %d -> %d", pid, replaced)
	}

	if stopped, err := s.Stop(&stderr); err != nil || !stopped {
		t.Errorf("Stop: %v, %v", stopped, err)
	}
	if state, _ := s.Status(ctx); state != Stopped || alive(replaced) {
		t.Errorf("after stop: %s", state)
	}
	for _, f := range []string{s.socket(), s.pidFile()} {
		if _, err := os.Stat(f); err == nil {
			t.Errorf("%s was left behind", f)
		}
	}
	if stopped, _ := s.Stop(&stderr); stopped {
		t.Error("second Stop reported a running server")
	}
}

// Two runs that need the server at the same time must end up sharing one.
func TestConcurrentEnsure(t *testing.T) {
	s, bin, cfg := setup(t)
	errs := make(chan error)
	for range 2 {
		go func() {
			_, err := s.Ensure(context.Background(), bin, cfg, &bytes.Buffer{})
			errs <- err
		}()
	}
	for range 2 {
		if err := <-errs; err != nil {
			t.Error(err)
		}
	}
	_, pid := s.Status(context.Background())
	var found []string
	files, _ := filepath.Glob("/proc/[0-9]*/cmdline")
	for _, f := range files {
		if data, _ := os.ReadFile(f); slices.Contains(strings.Split(string(data), "\x00"), s.socket()) {
			found = append(found, filepath.Base(filepath.Dir(f)))
		}
	}
	if len(found) != 1 || found[0] != strconv.Itoa(pid) {
		t.Errorf("want only server %d running, found %v", pid, found)
	}
}

func TestStartupFailure(t *testing.T) {
	s, bin, cfg := setup(t)
	t.Setenv(llamatest.FailEnv, "error: invalid argument: --sleep-idle-seconds")
	_, err := s.Ensure(context.Background(), bin, cfg, &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "invalid argument: --sleep-idle-seconds") {
		t.Errorf("want the server output in the error, got %v", err)
	}
}

// A PID file left by something else must never make easycommit use, or
// kill, a process that it did not start.
func TestForeignProcessIsIgnored(t *testing.T) {
	s, _, _ := setup(t)
	os.WriteFile(s.pidFile(), []byte(strconv.Itoa(os.Getpid())), 0o600)
	if state, _ := s.Status(context.Background()); state != Stopped {
		t.Errorf("foreign PID taken for the server: %s", state)
	}
	if stopped, _ := s.Stop(&bytes.Buffer{}); stopped {
		t.Error("Stop acted on a foreign PID")
	}
}

func TestArgs(t *testing.T) {
	s := &Server{Dir: "/run/ec"}
	cfg := config.Default()
	cfg.VRAMMB, cfg.CtxSize, cfg.GPULayers = 8192, 8192, 20
	got := strings.Join(s.args(cfg, []string{"--model", "m.gguf"}), " ")
	want := "--host /run/ec/server.sock --ctx-size 8192 --gpu-layers 20 --sleep-idle-seconds 600 --parallel 1 --model m.gguf"
	if got != want {
		t.Errorf("GPU:\n got %s\nwant %s", got, want)
	}
	cfg.CPU = true
	if got := s.args(cfg.WithProfile(), nil); !hasSuffix(got, []string{"--gpu-layers", "0", "--sleep-idle-seconds", "600", "--parallel", "1", "--device", "none"}) {
		t.Errorf("CPU: %v", got)
	}
}

func TestModelArgs(t *testing.T) {
	t.Setenv("HOME", "/home/someone")
	if got := modelArgs("~/models/m.gguf"); !slices.Equal(got, []string{"--model", "/home/someone/models/m.gguf"}) {
		t.Errorf("local file: %v", got)
	}
	if got := modelArgs("org/Name-GGUF:Q4_K_M"); !slices.Equal(got, []string{"--hf-repo", "org/Name-GGUF:Q4_K_M", "--no-mmproj"}) {
		t.Errorf("repository: %v", got)
	}
}

func TestDetectVRAM(t *testing.T) {
	bin := llamatest.Install(t)
	for devices, want := range map[string]int{
		llamatest.OneGPU: 16304,
		llamatest.NoGPU:  0,
		"0.00.001 I srv  llama_server: initializing ...\nAvailable devices:\n" +
			"  Vulkan0: Intel(R) UHD Graphics (7812 MiB, 7000 MiB free)\n" +
			"  Vulkan1: NVIDIA GeForce RTX 3060 (12288 MiB, 11000 MiB free)\n": 12288,
	} {
		t.Setenv(llamatest.DevicesEnv, devices)
		if got, err := DetectVRAM(context.Background(), bin); err != nil || got != want {
			t.Errorf("DetectVRAM = %d, %v; want %d for:\n%s", got, err, want, devices)
		}
	}
}

func TestFindBinary(t *testing.T) {
	_, err := FindBinary("llama-server-that-does-not-exist")
	if err == nil || !strings.Contains(err.Error(), "pacman -S llama-cpp ggml-vulkan") || !strings.Contains(err.Error(), "apk add") {
		t.Errorf("want install instructions, got %v", err)
	}
}
