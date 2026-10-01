// Package cli is the easycommit command line: it parses flags, wires the
// other packages together and prints the result.
package cli

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"unicode/utf8"

	"github.com/franelfers/easycommit/internal/config"
	"github.com/franelfers/easycommit/internal/gitdiff"
	"github.com/franelfers/easycommit/internal/llama"
	"github.com/franelfers/easycommit/internal/propose"
	"github.com/franelfers/easycommit/internal/server"
)

// Version is set at build time through -ldflags.
var Version = "dev"

const usage = `Usage:
  easycommit [flags]                 propose commit messages for the working tree
  easycommit server status [flags]   show whether the llama-server is running
  easycommit server stop [flags]     stop the llama-server

Flags:
`

// debugEnv, when set, prints every prompt and model answer to stderr.
const debugEnv = "EASYCOMMIT_DEBUG"

// errUsage marks errors caused by a wrong invocation.
var errUsage = errors.New("usage error")

type options struct {
	config.Config
	configPath string
	json       bool
	version    bool
}

// Run executes easycommit and returns its exit code. Proposals are the only
// thing written to stdout; progress, warnings and errors go to stderr.
func Run(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	err := run(ctx, args, stdout, stderr)
	switch {
	case err == nil, errors.Is(err, flag.ErrHelp):
		return 0
	case errors.Is(err, errUsage):
		return 2
	}
	fmt.Fprintln(stderr, "easycommit:", err)
	return 1
}

func run(ctx context.Context, args []string, stdout, stderr io.Writer) error {
	o, command, err := parse(args, stderr)
	if err != nil {
		return err
	}
	if o.version {
		fmt.Fprintln(stdout, "easycommit", Version)
		return nil
	}
	if err := o.Validate(); err != nil {
		return err
	}
	if len(command) == 0 {
		return generate(ctx, o, stdout, stderr)
	}

	srv, err := server.New()
	if err != nil {
		return err
	}
	switch strings.Join(command, " ") {
	case "server status":
		state, pid := srv.Status(ctx)
		if pid != 0 {
			fmt.Fprintf(stdout, "%s (pid %d)\n", state, pid)
		} else {
			fmt.Fprintln(stdout, state)
		}
	case "server stop":
		stopped, err := srv.Stop(stderr)
		if err != nil {
			return err
		}
		if stopped {
			fmt.Fprintln(stdout, "stopped")
		} else {
			fmt.Fprintln(stdout, server.Stopped)
		}
	default:
		fmt.Fprintf(stderr, "easycommit: unknown command %q\n", strings.Join(command, " "))
		return errUsage
	}
	return nil
}

// parse resolves the options with the precedence defaults < file < flags and
// returns the command words. Flags may come before or after them.
func parse(args []string, stderr io.Writer) (options, []string, error) {
	path, err := config.DefaultPath()
	if err != nil {
		return options{}, nil, err
	}
	// First pass: only to learn which configuration file to read. Its errors
	// are ignored because the second pass finds and reports them again.
	first := options{Config: config.Default(), configPath: path}
	parseArgs(&first, args, io.Discard)

	// Second pass: flags override what the file set.
	o := options{configPath: first.configPath}
	if o.Config, err = config.Load(o.configPath); err != nil {
		return options{}, nil, err
	}
	command, err := parseArgs(&o, args, stderr)
	return o, command, err
}

// parseArgs applies the flags in args to o, whose current values are the
// defaults, and returns the remaining words.
func parseArgs(o *options, args []string, stderr io.Writer) ([]string, error) {
	fs := flag.NewFlagSet("easycommit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() {
		fmt.Fprint(stderr, usage)
		fs.VisitAll(func(f *flag.Flag) {
			arg, text := flag.UnquoteUsage(f)
			if f.DefValue != "false" {
				text += fmt.Sprintf(" (default %s)", f.DefValue)
			}
			fmt.Fprintf(stderr, "  %s\n        %s\n", strings.TrimSpace("--"+f.Name+" "+arg), text)
		})
	}
	fs.StringVar(&o.configPath, "config", o.configPath, "configuration `file`")
	fs.StringVar(&o.ServerBin, "server-bin", o.ServerBin, "llama-server `binary`")
	fs.StringVar(&o.Model, "model", o.Model, "Hugging Face `repository` or path to a .gguf file")
	fs.IntVar(&o.CtxSize, "ctx-size", o.CtxSize, "context size in `tokens` (-1: from the VRAM profile)")
	fs.IntVar(&o.GPULayers, "gpu-layers", o.GPULayers, "model `layers` loaded on the GPU (-1: from the VRAM profile)")
	fs.BoolVar(&o.CPU, "cpu", o.CPU, "force inference on the CPU")
	fs.StringVar(&o.Body, "body", o.Body, "commit body `mode`: "+strings.Join(config.BodyModes, ", "))
	fs.StringVar(&o.Lang, "lang", o.Lang, "`language` of the messages")
	fs.IntVar(&o.IdleTimeout, "idle-timeout", o.IdleTimeout, "idle `seconds` before the model is unloaded")
	fs.BoolVar(&o.json, "json", false, "print the proposals as JSON")
	fs.BoolVar(&o.version, "version", false, "print the version")

	// Flags can sit before, between and after the command words.
	var command []string
	for {
		if err := fs.Parse(args); errors.Is(err, flag.ErrHelp) {
			return nil, err
		} else if err != nil {
			return nil, errUsage
		}
		if fs.NArg() == 0 {
			return command, nil
		}
		command, args = append(command, fs.Arg(0)), fs.Args()[1:]
	}
}

// generate is the main command: it prints proposals for the working tree.
func generate(ctx context.Context, o options, stdout, stderr io.Writer) error {
	u := newUI(stderr)
	defer u.clear()
	changes, err := gitdiff.Collect(ctx, ".", o.Exclude)
	if err != nil {
		return err
	}
	if len(changes) == 0 {
		u.note("no changes in the working tree")
		return nil
	}

	bin, err := server.FindBinary(o.ServerBin)
	if err != nil {
		return err
	}
	if o.VRAMMB == config.Unset && !o.CPU {
		if o.VRAMMB, err = server.DetectVRAM(ctx, bin); err != nil {
			return fmt.Errorf("detecting VRAM with %s --list-devices: %w", bin, err)
		}
		p := config.ProfileFor(o.VRAMMB)
		u.note("detected %d MiB of VRAM: using a context of %d tokens", o.VRAMMB, p.CtxSize)
		if err := config.SaveVRAM(o.configPath, o.VRAMMB); err != nil {
			u.note("warning: could not save vram_mb: %v", err)
		} else {
			u.note("saved vram_mb to %s", o.configPath)
		}
	}
	if !o.CPU && o.VRAMMB == 0 {
		u.note("warning: no usable GPU found, running inference on the CPU")
	}
	o.Config = o.WithProfile()

	srv, err := server.New()
	if err != nil {
		return err
	}
	var client *llama.Client
	err = u.wait("loading model", func() error {
		if client, err = srv.Ensure(ctx, bin, o.Config, u); err != nil {
			return err
		}
		// A server that unloaded the model while idle loads it again on the
		// first request that needs it: make that happen here.
		_, err = client.CountTokens(ctx, "")
		return err
	})
	if err != nil {
		return err
	}
	var model propose.Model = client
	if os.Getenv(debugEnv) != "" {
		model = traced{model, u}
	}
	proposal, err := propose.Generate(ctx, model, changes, propose.Options{
		CtxSize: o.CtxSize, Body: o.Body, Lang: o.Lang, Note: u.note, Status: u.status,
	})
	u.clear()
	if err != nil {
		return err
	}
	if o.json {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		enc.SetEscapeHTML(false)
		return enc.Encode(proposal)
	}
	_, err = io.WriteString(stdout, text(proposal))
	return err
}

// traced is a Model that dumps every inference, for prompt development.
type traced struct {
	propose.Model
	w io.Writer
}

func (t traced) Chat(ctx context.Context, req llama.ChatRequest) (string, error) {
	text, err := t.Model.Chat(ctx, req)
	fmt.Fprintf(t.w, "--- system\n%s\n--- user\n%s\n--- answer (error: %v)\n%s\n---\n", req.System, req.User, err, text)
	return text, err
}

// text renders a proposal in the default human-readable format: the
// all-in-one message, then every proposed commit with the tree of its files.
func text(p propose.Proposal) string {
	var b strings.Builder
	message := func(m propose.Message) {
		fmt.Fprintf(&b, "\n\"%s\"\n", m.Subject)
		if m.Body != nil {
			b.WriteString(*m.Body + "\n")
		}
	}
	b.WriteString(header("All In One"))
	message(p.AllInOne)

	noun := "commits"
	if len(p.Groups) == 1 {
		noun = "commit"
	}
	b.WriteString("\n" + header(fmt.Sprintf("Divided in %d %s", len(p.Groups), noun)))
	for _, g := range p.Groups {
		message(g.Message)
		for i, f := range g.Files {
			branch := "├─"
			if i == len(g.Files)-1 {
				branch = "└─"
			}
			fmt.Fprintf(&b, "%s %s\n", branch, f)
		}
	}
	return b.String()
}

// headerWidth is the length of the rules that open each section.
const headerWidth = 60

// header returns a title line padded with rule marks up to headerWidth columns.
func header(title string) string {
	line := "=== " + title + " "
	return line + strings.Repeat("=", max(3, headerWidth-utf8.RuneCountInString(line))) + "\n"
}
