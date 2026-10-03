package gitdiff

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// repo is a temporary Git repository for a test.
type repo struct {
	t   *testing.T
	dir string
}

func newRepo(t *testing.T) repo {
	t.Helper()
	r := repo{t, t.TempDir()}
	r.git("init", "-q", "-b", "main")
	r.git("config", "user.email", "test@example.com")
	r.git("config", "user.name", "Test")
	return r
}

func (r repo) git(args ...string) string {
	r.t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = r.dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		r.t.Fatalf("git %v: %v\n%s", args, err, out)
	}
	return string(out)
}

func (r repo) write(name, content string) {
	r.t.Helper()
	path := filepath.Join(r.dir, name)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		r.t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		r.t.Fatal(err)
	}
}

func (r repo) collect(dir string, exclude ...string) map[string]Change {
	r.t.Helper()
	changes, err := Collect(context.Background(), filepath.Join(r.dir, dir), exclude)
	if err != nil {
		r.t.Fatal(err)
	}
	byPath := map[string]Change{}
	for _, c := range changes {
		byPath[c.Path] = c
	}
	if len(byPath) != len(changes) {
		r.t.Fatalf("a path is reported twice: %+v", changes)
	}
	return byPath
}

func TestCollect(t *testing.T) {
	r := newRepo(t)
	r.write("staged.go", "package a\n")
	r.write("unstaged.go", "package a\n")
	r.write("both.go", "package a\n")
	r.write("gone.go", "package a\n\nfunc Removed() {}\n")
	r.write("sub/keep.go", "package sub\n")
	r.write(".gitignore", "ignored.log\n")
	r.git("add", "-A")
	r.git("commit", "-q", "-m", "init")

	r.write("staged.go", "package a\n\nfunc Staged() {}\n")
	r.write("both.go", "package a\n\nfunc First() {}\n")
	r.git("add", "staged.go", "both.go")
	r.write("both.go", "package a\n\nfunc First() {}\n\nfunc Second() {}\n")
	r.write("unstaged.go", "package a\n\nfunc Unstaged() {}\n")
	r.git("rm", "-q", "gone.go")
	r.write("new file.go", "package a\n\nfunc Untracked() {}\n")
	r.write("image.png", "\x89PNG\x00\x00binary\x00")
	r.write("vendor/lib/lib.go", "package lib\n")
	r.write("web/app.min.js", "var a=1;\n")
	r.write("go.sum", "example.com/mod v1.0.0 h1:abc=\n")
	r.write("ignored.log", "noise\n")
	before := r.git("status", "--porcelain")

	// Running from a subdirectory still covers the whole working tree.
	got := r.collect("sub", "go.sum", "vendor/", "*.min.js")

	want := map[string]struct{ status, omitted, contains string }{
		"staged.go":   {"modified", "", "+func Staged() {}"},
		"unstaged.go": {"modified", "", "+func Unstaged() {}"},
		"both.go":     {"modified", "", "+func Second() {}"},
		"gone.go":     {"deleted", "", "deleted file mode"},
		"new file.go": {"added", "", "+func Untracked() {}"},
		"image.png":   {"added", Binary, ""},
	}
	for path, w := range want {
		c, ok := got[path]
		if !ok {
			t.Errorf("%s: missing", path)
			continue
		}
		if c.Status != w.status || c.Omitted != w.omitted {
			t.Errorf("%s: status %q omitted %q, want %q %q", path, c.Status, c.Omitted, w.status, w.omitted)
		}
		if (c.Diff == "") != (w.omitted != "") || !strings.Contains(c.Diff, w.contains) {
			t.Errorf("%s: unexpected diff %q", path, c.Diff)
		}
	}
	if len(got) != len(want) {
		t.Errorf("got %d changes, want %d: %v", len(got), len(want), got)
	}
	if strings.Contains(got["both.go"].Diff, "-func First") || !strings.Contains(got["both.go"].Diff, "+func First() {}") {
		t.Errorf("staged and unstaged edits are not combined: %q", got["both.go"].Diff)
	}
	if strings.Contains(got["gone.go"].Diff, "Removed") {
		t.Errorf("deleted content leaked into the diff: %q", got["gone.go"].Diff)
	}
	if after := r.git("status", "--porcelain"); after != before {
		t.Errorf("Collect changed the repository:\n%s\nbefore:\n%s", after, before)
	}
}

func TestCollectEdgeCases(t *testing.T) {
	t.Run("clean tree has no changes", func(t *testing.T) {
		r := newRepo(t)
		r.write("a.go", "package a\n")
		r.git("add", "-A")
		r.git("commit", "-q", "-m", "init")
		if got := r.collect("."); len(got) != 0 {
			t.Errorf("want no changes, got %v", got)
		}
	})
	t.Run("repository without commits", func(t *testing.T) {
		r := newRepo(t)
		if got := r.collect("."); len(got) != 0 {
			t.Errorf("empty repository: want no changes, got %v", got)
		}
		r.write("staged.go", "package a\n")
		r.git("add", "staged.go")
		r.write("untracked.go", "package a\n")
		got := r.collect(".")
		if len(got) != 2 || got["staged.go"].Status != "added" || got["untracked.go"].Status != "added" {
			t.Errorf("unexpected changes: %v", got)
		}
	})
	t.Run("outside a repository", func(t *testing.T) {
		_, err := Collect(context.Background(), t.TempDir(), nil)
		if !errors.Is(err, ErrNotRepo) {
			t.Errorf("want ErrNotRepo, got %v", err)
		}
	})
}

func TestExcluded(t *testing.T) {
	patterns := []string{"go.sum", "vendor/", "*.min.js", "docs/*.gen.md"}
	for file, want := range map[string]bool{
		"go.sum":                true,
		"tools/go.sum":          true,
		"go.summary":            false,
		"vendor/x/y.go":         true,
		"pkg/vendor/y.go":       true,
		"vendors/y.go":          false,
		"vendor":                false,
		"web/app.min.js":        true,
		"web/app.js":            false,
		"docs/api.gen.md":       true,
		"other/docs/api.gen.md": false,
	} {
		if got := excluded(file, patterns); got != want {
			t.Errorf("excluded(%q) = %v, want %v", file, got, want)
		}
	}
}

func TestSplitDiff(t *testing.T) {
	a := "diff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -1 +1 @@\n-x\n+diff --git is not a header here\n"
	b := "diff --git a/b b/b\ndeleted file mode 100644\n"
	if got := splitDiff(a + b); !reflect.DeepEqual(got, []string{a, b}) {
		t.Errorf("got %q", got)
	}
	if got := splitDiff(""); got != nil {
		t.Errorf("empty diff: got %q", got)
	}
}

func TestCollectRenames(t *testing.T) {
	r := newRepo(t)
	r.write("old.md", "same\n")
	r.write("edited.md", "one\ntwo\nthree\nfour\n")
	r.git("add", "-A")
	r.git("commit", "-q", "-m", "init")
	r.git("mv", "old.md", "dir-new.md")
	r.git("mv", "edited.md", "edited2.md")
	r.write("edited2.md", "one\ntwo\nthree\nfive\n")

	got := r.collect(".")
	c := got["dir-new.md"]
	if c.Status != "renamed" || c.From != "old.md" || c.Omitted != Renamed || c.Diff != "" {
		t.Errorf("exact rename: %+v", c)
	}
	// A rename with edits stays as delete + add so the edits are shown.
	if got["edited.md"].Status != "deleted" || !strings.Contains(got["edited2.md"].Diff, "+five") || len(got) != 3 {
		t.Errorf("rename with edits: %+v", got)
	}
	if len(r.collect(".", "old.md")) != 2 {
		t.Error("excluding the old path should drop the rename")
	}
}
