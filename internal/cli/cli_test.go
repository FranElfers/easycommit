package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/franelfers/easycommit/internal/config"
	"github.com/franelfers/easycommit/internal/llama"
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
	want := `=== All In One =============================================

"chore: update 2 files"

=== Divided in 1 commit ====================================

"chore: update 2 files"
├─ a.go
└─ b.go
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
	if stdout, _, _ = e.run(); !strings.Contains(stdout, "\"chore: update 2 files\"\nBecause the tests say so.\n") {
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
	if code != 0 || !strings.HasPrefix(stdout, "=== All In One ===") {
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
	if stdout, _, code := e.run("server", "status"); code != 0 || !strings.HasPrefix(stdout, "running, model loaded (pid ") {
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
	body := "Explain why.\n- first item\n- second item"
	p := propose.Proposal{
		AllInOne: propose.Message{Subject: "feat: all", Body: &body},
		Groups: []propose.Group{
			{Name: "First", Files: []string{"a.go", "dir/b.go", "z.go"}, Message: propose.Message{Subject: "feat(a): one"}},
			{Name: "Second", Files: []string{"c.go"}, Message: propose.Message{Subject: "fix: two", Body: &body}},
		},
	}
	want := `=== All In One =============================================

"feat: all"
Explain why.
- first item
- second item

=== Divided in 2 commits ===================================

"feat(a): one"
├─ a.go
├─ dir/b.go
└─ z.go

"fix: two"
Explain why.
- first item
- second item
└─ c.go
`
	if got := text(p); got != want {
		t.Errorf("got:\n%s", got)
	}
}

func TestUIOnTerminal(t *testing.T) {
	var out bytes.Buffer
	u := &ui{w: &out, tty: true, refresh: time.Millisecond}
	// next returns what was written since the last call, with the control
	// sequences made visible.
	next := func() string {
		s := strings.NewReplacer("\r", "<CR>", "\x1b[K", "<EL>").Replace(out.String())
		out.Reset()
		return s
	}
	step := propose.Status{Step: 2, Steps: 3, Label: "summarizing 5 files"}

	u.status(step)
	if got := next(); got != "<CR>step 2 of 3: summarizing 5 files<EL>" {
		t.Errorf("step start: %q", got)
	}
	step.Progress = llama.Progress{PromptDone: 250, PromptTotal: 1000}
	u.status(step)
	if got := next(); got != "<CR>[#####---------------]  25% reading changes (step 2 of 3: summarizing 5 files)<EL>" {
		t.Errorf("reading: %q", got)
	}
	step.Progress = llama.Progress{PromptDone: 1000, PromptTotal: 1000, Generated: 42}
	u.status(step)
	if got := next(); got != "<CR>writing answer, 42 tokens (step 2 of 3: summarizing 5 files)<EL>" {
		t.Errorf("writing: %q", got)
	}

	// Updates within the same phase are drawn once per refresh interval.
	u.refresh = time.Hour
	step.Generated = 43
	u.status(step)
	if got := next(); got != "" {
		t.Errorf("update before the interval elapsed: %q", got)
	}
	u.refresh = 0
	u.status(step)
	if got := next(); got != "<CR>writing answer, 43 tokens (step 2 of 3: summarizing 5 files)<EL>" {
		t.Errorf("update after the interval elapsed: %q", got)
	}
	u.refresh = time.Millisecond

	// Anything else erases the live line first; clearing twice does nothing.
	u.note("something %s", "happened")
	if got := next(); got != "<CR><EL>easycommit: something happened\n" {
		t.Errorf("note: %q", got)
	}
	u.clear()
	if got := next(); got != "" {
		t.Errorf("clear without a live line: %q", got)
	}

	// A single step needs no numbering, and a long line is cut, not wrapped.
	u.status(propose.Status{Step: 1, Steps: 1, Label: "summarizing " + strings.Repeat("x", 100)})
	if got := next(); utf8.RuneCountInString(got) != len("<CR>")+liveWidth+len("<EL>") || !strings.HasSuffix(got, "x…<EL>") || strings.Contains(got, "step") {
		t.Errorf("long line: %q", got)
	}
	u.clear()
	next()

	// wait shows the elapsed time while nothing else is printed, yields to
	// other output, and leaves a clean line behind.
	err := u.wait("loading model", func() error {
		time.Sleep(20 * time.Millisecond)
		return errors.New("failed")
	})
	if got := next(); err == nil || !strings.HasPrefix(got, "<CR>loading model… 0 s<EL>") || !strings.HasSuffix(got, "<CR><EL>") {
		t.Errorf("wait: %v, %q", err, got)
	}
}

func TestUIOffTerminal(t *testing.T) {
	var out bytes.Buffer
	u := newUI(&out)
	step := propose.Status{Step: 1, Steps: 2, Label: "summarizing 5 files"}
	u.status(step)
	step.Progress = llama.Progress{PromptDone: 5, PromptTotal: 10}
	u.status(step)
	u.status(propose.Status{Step: 2, Steps: 2, Label: "proposing commits for 5 files"})
	u.wait("loading model", func() error { return nil })
	u.clear()
	want := "easycommit: step 1 of 2: summarizing 5 files\neasycommit: step 2 of 2: proposing commits for 5 files\n"
	if out.String() != want {
		t.Errorf("got %q", out.String())
	}
}

// TestRealServer runs the whole program against a real llama-server and the
// configured model. It is opt-in: set EASYCOMMIT_TEST_SERVER_BIN to the binary.
func TestRealServer(t *testing.T) {
	bin := os.Getenv("EASYCOMMIT_TEST_SERVER_BIN")
	if bin == "" {
		t.Skip("EASYCOMMIT_TEST_SERVER_BIN is not set")
	}
	e := newEnv(t)
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

func TestExcludeFlagAppends(t *testing.T) {
	o := options{Config: config.Default()}
	n := len(o.Exclude)
	if _, err := parseArgs(&o, []string{"--exclude", "a.yml", "--exclude", "b/"}, io.Discard); err != nil {
		t.Fatal(err)
	}
	if len(o.Exclude) != n+2 || o.Exclude[n] != "a.yml" || o.Exclude[n+1] != "b/" {
		t.Errorf("Exclude = %v", o.Exclude)
	}
}

func TestRenameNeedsNoModel(t *testing.T) {
	e := newEnv(t)
	e.write("old.md", "same\n")
	e.git("add", ".")
	e.git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
	e.git("mv", "old.md", "new.md")
	stdout, stderr, code := e.run()
	if code != 0 || !strings.Contains(stdout, `"chore: rename old.md to new.md"`) || strings.Count(stdout, "new.md") != 3 {
		t.Errorf("exit %d, stdout %q, stderr %q", code, stdout, stderr)
	}
	if stdout, _, _ := e.run("server", "status"); stdout != "not running\n" {
		t.Errorf("a rename started the server: %q", stdout)
	}
}
