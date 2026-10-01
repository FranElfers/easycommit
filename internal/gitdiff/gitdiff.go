// Package gitdiff reads every change of a Git working tree (staged, unstaged
// and untracked) without modifying the repository.
package gitdiff

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strings"
)

// ErrNotRepo is returned when the directory is not inside a Git working tree.
var ErrNotRepo = errors.New("not inside a Git working tree")

// Reasons why a Change carries no diff.
const (
	Binary   = "binary"
	Excluded = "excluded"
)

// Change is one file that differs from the last commit.
type Change struct {
	Path    string
	Status  string // added, modified, deleted or type changed
	Diff    string // unified diff; empty when Omitted is set
	Omitted string // Binary or Excluded: only path and status are reported
}

var statusNames = map[byte]string{'A': "added", 'D': "deleted", 'T': "type changed"}

// Collect returns the changes of the working tree containing dir, compared
// with HEAD. Files matching an exclude pattern keep only their path.
func Collect(ctx context.Context, dir string, exclude []string) ([]Change, error) {
	root, err := git(ctx, dir, nil, "rev-parse", "--show-toplevel")
	if err != nil {
		if errors.Is(err, exec.ErrNotFound) {
			return nil, err
		}
		return nil, ErrNotRepo
	}
	root = strings.TrimSuffix(root, "\n")

	env, cleanup, err := scratchIndex(ctx, root)
	if err != nil {
		return nil, err
	}
	defer cleanup()

	// Untracked files become "intent to add" entries of the scratch index, so
	// one diff against the last commit covers every kind of change.
	if _, err := git(ctx, root, env, "add", "--intent-to-add", "--", "."); err != nil {
		return nil, err
	}
	base := "HEAD"
	if _, err := git(ctx, root, nil, "rev-parse", "--verify", "--quiet", "HEAD"); err != nil {
		// No commits yet: compare with the empty tree.
		if base, err = git(ctx, root, nil, "hash-object", "-t", "tree", "/dev/null"); err != nil {
			return nil, err
		}
		base = strings.TrimSpace(base)
	}

	// Renames stay as delete + add so every path is listed and can be staged
	// on its own; deleted content is dropped because the path says enough.
	// Fixed prefixes keep the output independent of the user's diff settings.
	diffArgs := []string{"diff", "--no-color", "--no-ext-diff", "--src-prefix=a/", "--dst-prefix=b/",
		"--no-renames", "--irreversible-delete"}
	names, err := git(ctx, root, env, append(diffArgs, "--name-status", "-z", base, "--")...)
	if err != nil {
		return nil, err
	}
	full, err := git(ctx, root, env, append(diffArgs, base, "--")...)
	if err != nil {
		return nil, err
	}

	if names == "" {
		return nil, nil
	}
	fields := strings.Split(strings.TrimSuffix(names, "\x00"), "\x00")
	diffs := splitDiff(full)
	if len(fields)%2 != 0 || len(diffs) != len(fields)/2 {
		return nil, fmt.Errorf("unexpected git diff output: %d name fields, %d file diffs", len(fields), len(diffs))
	}

	changes := make([]Change, len(diffs))
	for i, diff := range diffs {
		c := Change{Path: fields[2*i+1], Status: "modified"}
		if name, ok := statusNames[fields[2*i][0]]; ok {
			c.Status = name
		}
		switch {
		case excluded(c.Path, exclude):
			c.Omitted = Excluded
		case isBinary(diff):
			c.Omitted = Binary
		default:
			c.Diff = diff
		}
		changes[i] = c
	}
	return changes, nil
}

// scratchIndex copies the repository index to a temporary file and returns
// the environment that makes git use it, keeping the real index untouched.
func scratchIndex(ctx context.Context, root string) (env []string, cleanup func(), err error) {
	src, err := git(ctx, root, nil, "rev-parse", "--git-path", "index")
	if err != nil {
		return nil, nil, err
	}
	src = strings.TrimSuffix(src, "\n")
	if !filepath.IsAbs(src) {
		src = filepath.Join(root, src)
	}

	tmp, err := os.CreateTemp("", "easycommit-index-*")
	if err != nil {
		return nil, nil, err
	}
	cleanup = func() {
		os.Remove(tmp.Name())
		os.Remove(tmp.Name() + ".lock")
	}
	defer tmp.Close()

	in, err := os.Open(src)
	switch {
	case errors.Is(err, os.ErrNotExist):
		// A repository without an index yet: git must create the file itself.
		os.Remove(tmp.Name())
	case err != nil:
		cleanup()
		return nil, nil, err
	default:
		defer in.Close()
		if _, err := io.Copy(tmp, in); err != nil {
			cleanup()
			return nil, nil, err
		}
	}
	return []string{"GIT_INDEX_FILE=" + tmp.Name()}, cleanup, nil
}

// splitDiff cuts a multi-file diff into one diff per file. Only file headers
// start a line with "diff --git", so the cut is unambiguous.
func splitDiff(full string) []string {
	const header = "diff --git "
	var parts []string
	for full != "" {
		next := strings.Index(full[1:], "\n"+header)
		if next < 0 {
			parts = append(parts, full)
			break
		}
		parts = append(parts, full[:next+2])
		full = full[next+2:]
	}
	return parts
}

// isBinary reports whether git refused to produce a textual diff.
func isBinary(diff string) bool {
	return !strings.Contains(diff, "\n@@ ") && strings.Contains(diff, "\nBinary files ")
}

// excluded reports whether file matches one of the patterns:
//
//	"dir/"   any path component named dir
//	"a/*.x"  a pattern with a slash globs the whole path
//	"*.x"    any other pattern globs the base name
func excluded(file string, patterns []string) bool {
	for _, p := range patterns {
		var ok bool
		switch dir, isDir := strings.CutSuffix(p, "/"); {
		case isDir:
			ok = strings.HasPrefix(file, dir+"/") || strings.Contains(file, "/"+dir+"/")
		case strings.Contains(p, "/"):
			ok, _ = path.Match(p, file)
		default:
			ok, _ = path.Match(p, path.Base(file))
		}
		if ok {
			return true
		}
	}
	return false
}

// git runs a git command in dir and returns its standard output.
func git(ctx context.Context, dir string, env []string, args ...string) (string, error) {
	cmd := exec.CommandContext(ctx, "git", args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), append(env, "LC_ALL=C", "GIT_OPTIONAL_LOCKS=0")...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("git %s: %w: %s", args[0], err, strings.TrimSpace(stderr.String()))
	}
	return string(out), nil
}
