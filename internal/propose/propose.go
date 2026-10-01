// Package propose turns the changes of a working tree into commit message
// proposals by running inferences that fit the model's context window.
package propose

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"unicode/utf8"

	"github.com/franelfers/easycommit/internal/gitdiff"
	"github.com/franelfers/easycommit/internal/llama"
)

const (
	// templateMargin covers the tokens the chat template adds around a prompt.
	templateMargin = 64
	// The expected size of an answer, which is kept free in the context: a
	// proposal holds the all-in-one message plus, per file, its path and at
	// worst a group of its own; a block summary holds a path and two
	// sentences per file. Longer answers are cut and asked again.
	answerBase     = 400
	answerPerFile  = 140
	summaryPerFile = 120
	// summaryTokens caps the rolling summary of an oversized file.
	summaryTokens = 400
	// attempts is how many times an invalid answer is asked again.
	attempts = 3
)

// ErrTooLarge means the changes cannot be described within the context
// window even after summarizing each file.
var ErrTooLarge = errors.New("too many changed files for the context size; raise ctx_size (--ctx-size)")

// Model is the part of llama-server that proposals need.
type Model interface {
	CountTokens(ctx context.Context, text string) (int, error)
	Chat(ctx context.Context, req llama.ChatRequest) (string, error)
}

// Message is a commit message. Body is nil when the message has none.
type Message struct {
	Subject string  `json:"subject"`
	Body    *string `json:"body"`
}

// Group is a set of files proposed as one commit.
type Group struct {
	Name  string   `json:"name"`
	Files []string `json:"files"`
	Message
}

// Proposal is the result printed to the user.
type Proposal struct {
	AllInOne Message `json:"all_in_one"`
	Groups   []Group `json:"groups"`
}

// Options control how proposals are generated.
type Options struct {
	CtxSize int
	Body    string // auto, never or always
	Lang    string
	// Note receives messages worth a line of their own. Optional.
	Note func(format string, args ...any)
	// Status follows the inference in progress. Optional.
	Status func(Status)
}

// Status is the state of the work: which inference is running, out of how
// many, and how far it is.
type Status struct {
	Step, Steps int
	Label       string
	llama.Progress
}

type generator struct {
	Options
	model       Model
	step, steps int
}

// track announces the inference about to run as the current step and
// returns the callback that follows its progress.
func (g *generator) track(label string) func(llama.Progress) {
	report := func(p llama.Progress) {
		g.Status(Status{Step: g.step, Steps: g.steps, Label: label, Progress: p})
	}
	report(llama.Progress{})
	return report
}

// files words a number of files.
func files(n int) string {
	if n == 1 {
		return "1 file"
	}
	return fmt.Sprintf("%d files", n)
}

// Generate proposes commit messages for changes. When they fit in the context
// window a single inference reads the diffs; otherwise the diffs are
// summarized per file first and the final inference reads the summaries.
func Generate(ctx context.Context, model Model, changes []gitdiff.Change, o Options) (Proposal, error) {
	if o.Note == nil {
		o.Note = func(string, ...any) {}
	}
	if o.Status == nil {
		o.Status = func(Status) {}
	}
	g := generator{Options: o, model: model, steps: 1}
	language := fmt.Sprintf(languageRule, languageName(o.Lang))
	system := fmt.Sprintf(proposalPrompt, bodyRules[o.Body], language)

	paths := make([]string, len(changes))
	sections := make([]string, len(changes))
	for i, c := range changes {
		paths[i], sections[i] = c.Path, section(c, c.Diff)
	}
	pathTokens, err := model.CountTokens(ctx, strings.Join(paths, "\n"))
	if err != nil {
		return Proposal{}, err
	}
	answer := answerBase + len(paths)*answerPerFile + pathTokens
	room, err := g.room(ctx, system+language)
	if err != nil {
		return Proposal{}, err
	}
	room -= answer
	size, err := model.CountTokens(ctx, strings.Join(sections, "\n"))
	if err != nil {
		return Proposal{}, err
	}
	if size > room {
		g.Note("changes need %d tokens but %d fit in the context: summarizing each file first", size, max(room, 0))
		if sections, err = g.summarize(ctx, changes); err != nil {
			return Proposal{}, err
		}
		if size, err = model.CountTokens(ctx, strings.Join(sections, "\n")); err != nil {
			return Proposal{}, err
		}
		if size > room {
			return Proposal{}, ErrTooLarge
		}
	}

	g.step = g.steps
	req := llama.ChatRequest{
		System: system, User: strings.Join(sections, "\n") + language,
		Schema: proposalSchema(paths, o.Body), MaxTokens: answer,
		Progress: g.track("proposing commits for " + files(len(changes))),
	}
	return g.propose(ctx, req, paths)
}

// room is how many tokens are left in the context next to the system
// prompt, for the user content and the answer.
func (g *generator) room(ctx context.Context, system string) (int, error) {
	fixed, err := g.model.CountTokens(ctx, system)
	return g.CtxSize - fixed - templateMargin, err
}

// propose runs the final inference and turns its answer into a Proposal.
func (g *generator) propose(ctx context.Context, req llama.ChatRequest, paths []string) (Proposal, error) {
	type group struct {
		Name string `json:"name"`
		Message
	}
	type answer struct {
		Groups   []group           `json:"groups"`
		Files    map[string]string `json:"files"`
		AllInOne Message           `json:"all_in_one"`
	}
	a, err := ask(ctx, g, req, func(a answer) error {
		for _, p := range paths {
			named := func(g group) bool { return g.Name == a.Files[p] }
			if !slices.ContainsFunc(a.Groups, named) {
				return fmt.Errorf(`"files" puts %q in group %q, which "groups" does not have`, p, a.Files[p])
			}
		}
		if !subjectRE.MatchString(a.AllInOne.Subject) {
			return fmt.Errorf("subject %q is not a Conventional Commits subject", a.AllInOne.Subject)
		}
		for _, grp := range a.Groups {
			if !subjectRE.MatchString(grp.Subject) {
				return fmt.Errorf("subject %q is not a Conventional Commits subject", grp.Subject)
			}
		}
		return nil
	})
	if err != nil {
		return Proposal{}, err
	}

	p := Proposal{AllInOne: g.clean(a.AllInOne)}
	for i, grp := range a.Groups {
		out := Group{Name: grp.Name, Message: g.clean(grp.Message)}
		for _, path := range paths {
			if a.Files[path] == grp.Name {
				out.Files = append(out.Files, path)
			}
		}
		// A repeated name or a group left without files adds nothing.
		first := slices.IndexFunc(a.Groups, func(o group) bool { return o.Name == grp.Name }) == i
		if first && len(out.Files) > 0 {
			p.Groups = append(p.Groups, out)
		}
	}
	return p, nil
}

// clean normalizes a message from the model: a blank body counts as none.
func (g *generator) clean(m Message) Message {
	m.Subject = strings.TrimSpace(m.Subject)
	if m.Body != nil {
		if body := strings.TrimSpace(*m.Body); body == "" || g.Body == "never" {
			m.Body = nil
		} else {
			m.Body = &body
		}
	}
	return m
}

// summarize returns one section per change in which every diff is replaced
// by a summary. Diffs are packed into blocks that fit the context window; a
// file too large for a block of its own gets a rolling summary instead.
func (g *generator) summarize(ctx context.Context, changes []gitdiff.Change) ([]string, error) {
	room, err := g.room(ctx, summaryPrompt)
	if err != nil {
		return nil, err
	}
	var textual []gitdiff.Change
	var sizes []int
	for _, c := range changes {
		if c.Omitted != "" {
			continue
		}
		size, err := g.model.CountTokens(ctx, section(c, c.Diff))
		if err != nil {
			return nil, err
		}
		textual, sizes = append(textual, c), append(sizes, size)
	}

	// A file takes up its diff and, in the same context, its summary.
	costs := make([]int, len(sizes))
	for i, size := range sizes {
		costs[i] = size + summaryPerFile
	}

	// Every block is a step, and the proposal is the last one.
	summaries := map[string]string{}
	blocks := pack(costs, room)
	g.steps = len(blocks) + 1
	for i, block := range blocks {
		g.step = i + 1
		first := block[0]
		if costs[first] > room {
			if summaries[textual[first].Path], err = g.rolling(ctx, textual[first], sizes[first], room); err != nil {
				return nil, err
			}
			continue
		}
		answer, err := g.summarizeBlock(ctx, textual[first:first+len(block)])
		if err != nil {
			return nil, err
		}
		for path, summary := range answer {
			summaries[path] = summary
		}
	}

	sections := make([]string, len(changes))
	for i, c := range changes {
		sections[i] = section(c, "Summary: "+summaries[c.Path])
	}
	return sections, nil
}

// summarizeBlock asks for the summary of every file of a block at once.
func (g *generator) summarizeBlock(ctx context.Context, block []gitdiff.Change) (map[string]string, error) {
	paths := make([]string, len(block))
	sections := make([]string, len(block))
	for i, c := range block {
		paths[i], sections[i] = c.Path, section(c, c.Diff)
	}
	req := llama.ChatRequest{
		System:    summaryPrompt,
		User:      strings.Join(sections, "\n"),
		Schema:    perFile(paths, schema{"type": "string", "minLength": 1}),
		MaxTokens: len(block) * summaryPerFile,
		Progress:  g.track("summarizing " + files(len(block))),
	}
	return ask(ctx, g, req, func(answer map[string]string) error {
		for _, p := range paths {
			if strings.TrimSpace(answer[p]) == "" {
				return fmt.Errorf("the summary of %q is missing", p)
			}
		}
		return nil
	})
}

// rolling summarizes a file that exceeds the context window: it is read in
// consecutive parts, each one updating the summary of the previous ones.
// Every request holds a part, the summary so far and the new summary.
func (g *generator) rolling(ctx context.Context, c gitdiff.Change, size, room int) (string, error) {
	parts := split(c.Diff, size, room-2*summaryTokens-templateMargin)
	summary := "(none yet)"
	for i, part := range parts {
		req := llama.ChatRequest{
			System:    rollingPrompt,
			User:      fmt.Sprintf(rollingUser, c.Path, c.Status, i+1, len(parts), summary, part),
			MaxTokens: summaryTokens,
			Progress:  g.track(fmt.Sprintf("summarizing %s, part %d of %d", c.Path, i+1, len(parts))),
		}
		text, err := g.model.Chat(ctx, req)
		// A summary cut at the token limit is still a usable summary.
		if err != nil && !errors.Is(err, llama.ErrTruncated) {
			return "", err
		}
		summary = strings.TrimSpace(text)
	}
	return summary, nil
}

// ask runs an inference constrained by req.Schema and decodes the answer.
// An answer that is cut short or that check rejects is asked again with the
// reason appended.
func ask[T any](ctx context.Context, g *generator, req llama.ChatRequest, check func(T) error) (T, error) {
	var zero T
	var invalid error
	user := req.User
	for range attempts {
		if invalid != nil {
			g.Note("the model answer was rejected (%v), asking again", invalid)
			req.User = fmt.Sprintf("%s\n\nYour previous answer was rejected: %v. Answer again without that mistake.", user, invalid)
		}
		text, err := g.model.Chat(ctx, req)
		var answer T
		switch {
		case errors.Is(err, llama.ErrTruncated):
			invalid = errors.New("it was too long")
		case err != nil:
			return zero, err
		default:
			if invalid = json.Unmarshal([]byte(text), &answer); invalid == nil {
				invalid = check(answer)
			}
		}
		if invalid == nil {
			return answer, nil
		}
	}
	return zero, fmt.Errorf("the model answer is still invalid after %d attempts: %w", attempts, invalid)
}

// pack distributes consecutive items into blocks whose sizes add up to at
// most capacity. An item larger than capacity gets a block of its own.
func pack(sizes []int, capacity int) [][]int {
	var blocks [][]int
	used := 0
	for i, size := range sizes {
		if len(blocks) == 0 || used+size > capacity {
			blocks = append(blocks, nil)
			used = 0
		}
		blocks[len(blocks)-1] = append(blocks[len(blocks)-1], i)
		used += size
	}
	return blocks
}

// split cuts text, known to hold the given number of tokens, into parts of
// at most limit tokens each, preferring to cut at line ends. Sizes are
// estimated from the average bytes per token of the text, with a margin for
// stretches denser than the average.
func split(text string, tokens, limit int) []string {
	maxBytes := max(1, len(text)*max(limit, 1)/max(tokens, 1)*9/10)
	var parts []string
	for len(text) > maxBytes {
		cut := strings.LastIndexByte(text[:maxBytes], '\n') + 1
		if cut == 0 {
			// A single line longer than a part: cut it between characters.
			for cut = maxBytes; cut > 1 && !utf8.RuneStart(text[cut]); cut-- {
			}
		}
		parts = append(parts, text[:cut])
		text = text[cut:]
	}
	return append(parts, text)
}
