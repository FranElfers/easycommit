package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/franelfers/easycommit/internal/llamatest"
	"github.com/franelfers/easycommit/internal/propose"
)

func TestMain(m *testing.M) {
	llamatest.Main()
	os.Exit(m.Run())
}

// env is an isolated easycommit environment: a Git repository as working
// directory, a fake llama-server and private configuration and state.
type env struct {
	t      *testing.T
	config string
	base   []string
}

func newEnv(t *testing.T) *env {
	t.Helper()
	t.Setenv("XDG_RUNTIME_DIR", llamatest.ShortDir(t))
	t.Setenv("HF_HUB_CACHE", t.TempDir())
	t.Chdir(t.TempDir())
	e := &env{t: t, config: filepath.Join(t.TempDir(), "config.toml")}
	e.base = []string{"--config", e.config, "--server-bin", llamatest.Install(t)}
	e.git("init", "-q", "-b", "main")
	t.Cleanup(func() { e.run("server", "stop") })
	return e
}

func (e *env) git(args ...string) {
	e.t.Helper()
	if out, err := exec.Command("git", args...).CombinedOutput(); err != nil {
		e.t.Fatalf("git %v: %v\n%s", args, err, out)
	}
}

func (e *env) write(name, content string) {
	e.t.Helper()
	if err := os.WriteFile(name, []byte(content), 0o644); err != nil {
		e.t.Fatal(err)
	}
}

// run executes easycommit with the environment's flags followed by args.
func (e *env) run(args ...string) (stdout, stderr string, code int) {
	return e.exec(append(e.base[:len(e.base):len(e.base)], args...))
}

func (e *env) exec(args []string) (stdout, stderr string, code int) {
	var out, errs bytes.Buffer
	code = Run(context.Background(), args, &out, &errs)
	return out.String(), errs.String(), code
}

func TestProposals(t *testing.T) {
	e := newEnv(t)
	e.write("a.go", "package a\n")
	e.write("b.go", "package a\n")

	stdout, stderr, code := e.run()
	want := `All In One:
"chore: update 2 files"

Group "all changes":
  a.go
  b.go
"chore: update 2 files"
`
	if code != 0 || stdout != want {
		t.Fatalf("exit %d\nstdout:\n%s\nstderr:\n%s", code, stdout, stderr)
	}
	// First run: the VRAM is detected, reported and saved.
	if !strings.Contains(stderr, "detected 16304 MiB of VRAM") {
		t.Errorf("stderr lacks the detection notice:\n%s", stderr)
	}
	if saved, _ := os.ReadFile(e.config); string(saved) != "vram_mb = 16304\n" {
		t.Errorf("config file: %q", saved)
	}

	// Later runs neither detect again nor restart the server.
	stdout, stderr, code = e.run("--json", "--body", "always")
	if code != 0 || strings.Contains(stderr, "detected") || strings.Contains(stderr, "starting") {
		t.Fatalf("exit %d, stderr:\n%s", code, stderr)
	}
	var p propose.Proposal
	if err := json.Unmarshal([]byte(stdout), &p); err != nil {
		t.Fatalf("stdout is not the JSON format: %v\n%s", err, stdout)
	}
	if p.AllInOne.Body == nil || len(p.Groups) != 1 || len(p.Groups[0].Files) != 2 {
		t.Errorf("unexpected proposal: %s", stdout)
	}
	for _, key := range []string{`"all_in_one"`, `"groups"`, `"name"`, `"files"`, `"subject"`, `"body"`} {
		if !strings.Contains(stdout, key) {
			t.Errorf("JSON output lacks %s:\n%s", key, stdout)
		}
	}

	// The file sets the defaults and flags override it.
	e.write(e.config, "vram_mb = 16304\nbody = \"always\"\n")
	if stdout, _, _ = e.run(); !strings.Contains(stdout, "\"chore: update 2 files\n\nBecause the tests say so.\"\n") {
		t.Errorf("body from the config file is missing:\n%s", stdout)
	}
	if stdout, _, _ = e.run("--body", "never"); strings.Contains(stdout, "Because") {
		t.Errorf("flag did not override the file:\n%s", stdout)
	}
}

func TestNothingToDo(t *testing.T) {
	e := newEnv(t)
	stdout, stderr, code := e.run()
	if code != 0 || stdout != "" || !strings.Contains(stderr, "no changes") {
		t.Errorf("no changes: exit %d, stdout %q, stderr %q", code, stdout, stderr)
	}
	// Nothing was needed from llama-server, so it must not be running.
	if stdout, _, _ := e.run("server", "status"); stdout != "not running\n" {
		t.Errorf("server status: %q", stdout)
	}

	t.Chdir(t.TempDir())
	stdout, stderr, code = e.run()
	if code != 1 || stdout != "" || !strings.Contains(stderr, "not inside a Git working tree") {
		t.Errorf("outside a repository: exit %d, stdout %q, stderr %q", code, stdout, stderr)
	}
}

func TestCPUFallback(t *testing.T) {
	e := newEnv(t)
	e.write("a.go", "package a\n")
	t.Setenv(llamatest.DevicesEnv, llamatest.NoGPU)

	stdout, stderr, code := e.run()
	if code != 0 || !strings.HasPrefix(stdout, "All In One:\n") {
		t.Fatalf("exit %d\n%s\n%s", code, stdout, stderr)
	}
	for _, want := range []string{"warning: no usable GPU found", "CPU only"} {
		if !strings.Contains(stderr, want) {
			t.Errorf("stderr lacks %q:\n%s", want, stderr)
		}
	}
	// --cpu is a choice, not a fallback: there is nothing to warn about.
	if _, stderr, _ = e.run("--cpu"); strings.Contains(stderr, "warning") {
		t.Errorf("unexpected warning with --cpu:\n%s", stderr)
	}
}

func TestServerCommands(t *testing.T) {
	e := newEnv(t)
	e.write("a.go", "package a\n")
	if _, stderr, code := e.run(); code != 0 {
		t.Fatal(stderr)
	}
	// Flags are accepted after the command words too.
	if stdout, _, code := e.exec(append([]string{"server", "status"}, e.base...)); code != 0 || !strings.HasPrefix(stdout, "running, model loaded (pid ") {
		t.Errorf("status: exit %d, %q", code, stdout)
	}
	if stdout, _, code := e.run("server", "stop"); code != 0 || stdout != "stopped\n" {
		t.Errorf("stop: exit %d, %q", code, stdout)
	}
	if stdout, _, _ := e.run("server", "stop"); stdout != "not running\n" {
		t.Errorf("second stop: %q", stdout)
	}
}

func TestUsage(t *testing.T) {
	e := newEnv(t)
	for name, args := range map[string][]string{
		"unknown flag":    {"--nope"},
		"unknown command": {"commit"},
		"partial command": {"server"},
	} {
		if stdout, stderr, code := e.run(args...); code != 2 || stdout != "" || stderr == "" {
			t.Errorf("%s: exit %d, stdout %q, stderr %q", name, code, stdout, stderr)
		}
	}
	if _, stderr, code := e.run("--body", "sometimes"); code != 1 || !strings.Contains(stderr, "body must be one of") {
		t.Errorf("invalid value: exit %d, %q", code, stderr)
	}
	if _, stderr, code := e.run("--help"); code != 0 || !strings.Contains(stderr, "--ctx-size tokens") {
		t.Errorf("help: exit %d, %q", code, stderr)
	}
	if stdout, _, code := e.run("--version"); code != 0 || stdout != "easycommit dev\n" {
		t.Errorf("version: exit %d, %q", code, stdout)
	}
	e.write(e.config, "nope = 1\n")
	if _, stderr, code := e.run(); code != 1 || !strings.Contains(stderr, `unknown key "nope"`) {
		t.Errorf("bad config: exit %d, %q", code, stderr)
	}
}

func TestText(t *testing.T) {
	body := "Explain why.\nOn two lines."
	p := propose.Proposal{
		AllInOne: propose.Message{Subject: "feat: all", Body: &body},
		Groups: []propose.Group{
			{Name: "First", Files: []string{"a.go", "dir/b.go"}, Message: propose.Message{Subject: "feat(a): one"}},
			{Name: "Second", Files: []string{"c.go"}, Message: propose.Message{Subject: "fix: two", Body: &body}},
		},
	}
	want := `All In One:
"feat: all

Explain why.
On two lines."

Group "First":
  a.go
  dir/b.go
"feat(a): one"

Group "Second":
  c.go
"fix: two

Explain why.
On two lines."
`
	if got := text(p); got != want {
		t.Errorf("got:\n%s", got)
	}
}

// TestRealServer runs the whole program against a real llama-server and the
// configured model. It is opt-in: set EASYCOMMIT_TEST_SERVER_BIN to the binary.
func TestRealServer(t *testing.T) {
	bin := os.Getenv("EASYCOMMIT_TEST_SERVER_BIN")
	if bin == "" {
		t.Skip("EASYCOMMIT_TEST_SERVER_BIN is not set")
	}
	hub := os.Getenv("HF_HUB_CACHE")
	e := newEnv(t)
	t.Setenv("HF_HUB_CACHE", hub)
	e.base[3] = bin
	e.write("main.go", "package main\n\nfunc main() { println(\"hello\") }\n")
	e.write("README.md", "# Demo\n\nPrints a greeting.\n")

	stdout, stderr, code := e.run("--json")
	t.Logf("stderr:\n%s\nstdout:\n%s", stderr, stdout)
	if code != 0 {
		t.Fatalf("exit %d", code)
	}
	var p propose.Proposal
	if err := json.Unmarshal([]byte(stdout), &p); err != nil {
		t.Fatal(err)
	}
	files := 0
	for _, g := range p.Groups {
		files += len(g.Files)
	}
	if files != 2 || p.AllInOne.Subject == "" {
		t.Errorf("unexpected proposal: %+v", p)
	}
}
