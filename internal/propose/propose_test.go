package propose

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/franelfers/easycommit/internal/gitdiff"
	"github.com/franelfers/easycommit/internal/llama"
	"github.com/franelfers/easycommit/internal/llamatest"
)

// truncated, as a scripted answer, makes the fake model report a cut answer.
const truncated = "<truncated>"

// fakeModel answers like the fake llama-server and records every request.
// answers, when set, replaces the first answers in order.
type fakeModel struct {
	requests []llama.ChatRequest
	answers  []string
}

func (m *fakeModel) CountTokens(_ context.Context, text string) (int, error) {
	return llamatest.CountTokens(text), nil
}

func (m *fakeModel) Chat(_ context.Context, req llama.ChatRequest) (string, error) {
	m.requests = append(m.requests, req)
	if req.Progress != nil {
		req.Progress(llama.Progress{PromptDone: 7, PromptTotal: 7, Generated: 3})
	}
	if n := len(m.requests); n <= len(m.answers) {
		if m.answers[n-1] == truncated {
			return "{", llama.ErrTruncated
		}
		return m.answers[n-1], nil
	}
	schema, _ := req.Schema.(json.RawMessage)
	return llamatest.Answer(schema), nil
}

// kinds names each recorded request after the prompt that produced it.
func (m *fakeModel) kinds() string {
	names := map[string]string{summaryPrompt: "summary", rollingPrompt: "rolling"}
	var kinds []string
	for _, r := range m.requests {
		kind, ok := names[r.System]
		if !ok {
			kind = "proposal"
		}
		kinds = append(kinds, kind)
	}
	return strings.Join(kinds, " ")
}

// change builds a text change whose diff has the given number of words.
func change(path string, words int) gitdiff.Change {
	return gitdiff.Change{Path: path, Status: "modified", Diff: strings.Repeat("word\n", words)}
}

func generate(t *testing.T, m *fakeModel, ctxSize int, body string, changes ...gitdiff.Change) (Proposal, error) {
	t.Helper()
	return Generate(context.Background(), m, changes, Options{CtxSize: ctxSize, Body: body, Lang: "en"})
}

func TestSingleInference(t *testing.T) {
	m := &fakeModel{}
	binary := gitdiff.Change{Path: "logo.png", Status: "added", Omitted: gitdiff.Binary}
	p, err := generate(t, m, 8192, "auto", change("a.go", 100), change("b.go", 100), binary)
	if err != nil {
		t.Fatal(err)
	}
	if m.kinds() != "proposal" {
		t.Fatalf("inferences: %s", m.kinds())
	}
	want := Proposal{
		AllInOne: Message{Subject: "chore: update 3 files"},
		Groups: []Group{{
			Name: llamatest.GroupName, Files: []string{"a.go", "b.go", "logo.png"},
			Message: Message{Subject: "chore: update 3 files"},
		}},
	}
	if !reflect.DeepEqual(p, want) {
		t.Errorf("got %+v", p)
	}
	user := m.requests[0].User
	if !strings.Contains(user, "### a.go (modified)\nword\n") || !strings.Contains(user, "### logo.png (added, binary: content omitted)") {
		t.Errorf("unexpected prompt:\n%s", user)
	}
	// The answer limit grows with the number of files and their paths.
	if want := answerBase + 3*answerPerFile + 3; m.requests[0].MaxTokens != want {
		t.Errorf("answer limit: %d, want %d", m.requests[0].MaxTokens, want)
	}
}

func TestSummarizeInBlocks(t *testing.T) {
	// With a 4096-token context about 3900 tokens of diffs and summaries fit
	// in a request: the three files need two blocks, and the lock file
	// needs no summary.
	m := &fakeModel{}
	lock := gitdiff.Change{Path: "go.sum", Status: "modified", Omitted: gitdiff.Excluded}
	p, err := generate(t, m, 4096, "auto", change("a.go", 1500), change("b.go", 1000), lock, change("c.go", 1500))
	if err != nil {
		t.Fatal(err)
	}
	if m.kinds() != "summary summary proposal" {
		t.Fatalf("inferences: %s", m.kinds())
	}
	if first := m.requests[0].User; !strings.Contains(first, "### a.go") || !strings.Contains(first, "### b.go") || strings.Contains(first, "### c.go") {
		t.Errorf("first block should hold a.go and b.go only")
	}
	final := m.requests[2].User
	for _, want := range []string{"### a.go (modified)\nSummary: summary of a.go", "### c.go (modified)\nSummary: summary of c.go", "### go.sum (modified, excluded: content omitted)"} {
		if !strings.Contains(final, want) {
			t.Errorf("final prompt lacks %q:\n%s", want, final)
		}
	}
	if strings.Contains(final, "word") {
		t.Error("final prompt still carries diff content")
	}
	if len(p.Groups) != 1 || len(p.Groups[0].Files) != 4 {
		t.Errorf("every file must be in a group: %+v", p.Groups)
	}
}

func TestRollingSummary(t *testing.T) {
	// huge.go alone exceeds the context: it is read in parts, each request
	// carrying the summary of the previous ones.
	m := &fakeModel{answers: []string{"first summary", "second summary", "third summary"}}
	if _, err := generate(t, m, 4096, "auto", change("huge.go", 6000), change("small.go", 10)); err != nil {
		t.Fatal(err)
	}
	if m.kinds() != "rolling rolling rolling summary proposal" {
		t.Fatalf("inferences: %s", m.kinds())
	}
	for i, r := range m.requests[:3] {
		if llamatest.CountTokens(r.User)+llamatest.CountTokens(r.System) > 4096-r.MaxTokens {
			t.Errorf("part %d does not fit in the context", i+1)
		}
	}
	if !strings.Contains(m.requests[0].User, "part 1 of 3") || !strings.Contains(m.requests[1].User, "Summary so far:\nfirst summary") {
		t.Errorf("unexpected rolling prompts:\n%s", m.requests[1].User[:200])
	}
	if final := m.requests[4].User; !strings.Contains(final, "### huge.go (modified)\nSummary: third summary") {
		t.Errorf("final prompt lacks the rolling summary:\n%s", final)
	}
}

// Every inference is announced as a step before it runs and followed while
// it does.
func TestStatus(t *testing.T) {
	run := func(ctxSize int, changes ...gitdiff.Change) (seen []string) {
		t.Helper()
		status := func(s Status) {
			seen = append(seen, fmt.Sprintf("%d/%d %s +%d", s.Step, s.Steps, s.Label, s.Generated))
		}
		_, err := Generate(context.Background(), &fakeModel{}, changes, Options{CtxSize: ctxSize, Body: "auto", Lang: "en", Status: status})
		if err != nil {
			t.Fatal(err)
		}
		return seen
	}
	if got, want := run(8192, change("a.go", 10)), []string{
		"1/1 proposing commits for 1 file +0",
		"1/1 proposing commits for 1 file +3",
	}; !reflect.DeepEqual(got, want) {
		t.Errorf("single inference:\n%s", strings.Join(got, "\n"))
	}

	got := run(4096, change("huge.go", 6000), change("a.go", 1500), change("b.go", 1000), change("c.go", 1500))
	want := []string{
		"1/4 summarizing huge.go, part 1 of 3 +0", "1/4 summarizing huge.go, part 1 of 3 +3",
		"1/4 summarizing huge.go, part 2 of 3 +0", "1/4 summarizing huge.go, part 2 of 3 +3",
		"1/4 summarizing huge.go, part 3 of 3 +0", "1/4 summarizing huge.go, part 3 of 3 +3",
		"2/4 summarizing 2 files +0", "2/4 summarizing 2 files +3",
		"3/4 summarizing 1 file +0", "3/4 summarizing 1 file +3",
		"4/4 proposing commits for 4 files +0", "4/4 proposing commits for 4 files +3",
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("summaries first:\n%s", strings.Join(got, "\n"))
	}
}

func TestTooLarge(t *testing.T) {
	var changes []gitdiff.Change
	for i := range 2000 {
		changes = append(changes, gitdiff.Change{Path: fmt.Sprintf("img/%d.png", i), Status: "added", Omitted: gitdiff.Binary})
	}
	if _, err := generate(t, &fakeModel{}, 4096, "auto", changes...); !errors.Is(err, ErrTooLarge) {
		t.Errorf("want ErrTooLarge, got %v", err)
	}
}

func TestInvalidAnswers(t *testing.T) {
	const good = `{"files":{"a.go":"x"},"groups":[{"name":"x","subject":"fix: repair a","body":null}],"all_in_one":{"subject":"fix: repair a","body":null}}`
	bad := map[string]string{
		"unknown group": `{"files":{"a.go":"y"},"groups":[{"name":"x","subject":"fix: a","body":null}],"all_in_one":{"subject":"fix: a","body":null}}`,
		"missing file":  `{"files":{},"groups":[{"name":"x","subject":"fix: a","body":null}],"all_in_one":{"subject":"fix: a","body":null}}`,
		"bad subject":   `{"files":{"a.go":"x"},"groups":[{"name":"x","subject":"Fixed a.","body":null}],"all_in_one":{"subject":"fix: a","body":null}}`,
		"bad global":    `{"files":{"a.go":"x"},"groups":[{"name":"x","subject":"fix: a","body":null}],"all_in_one":{"subject":"fix it","body":null}}`,
		"not JSON":      `I think the commit should be...`,
		"cut short":     truncated,
	}
	for name, answer := range bad {
		// One bad answer is asked again, with the reason in the new prompt.
		m := &fakeModel{answers: []string{answer, good}}
		p, err := generate(t, m, 8192, "auto", change("a.go", 10))
		if err != nil || p.AllInOne.Subject != "fix: repair a" || len(m.requests) != 2 {
			t.Errorf("%s: no recovery: %+v, %v", name, p, err)
			continue
		}
		if !strings.Contains(m.requests[1].User, "Your previous answer was rejected") {
			t.Errorf("%s: retry prompt lacks the reason", name)
		}

		m = &fakeModel{answers: []string{answer, answer, answer, good}}
		if _, err := generate(t, m, 8192, "auto", change("a.go", 10)); err == nil || len(m.requests) != attempts {
			t.Errorf("%s: want failure after %d attempts, got %v after %d", name, attempts, err, len(m.requests))
		}
	}
}

func TestGroupsFromAnswer(t *testing.T) {
	// Groups without files and repeated names are dropped; files keep the
	// order of the changes; blank bodies become none.
	answer := `{"files":{"b.go":"two","a.go":"one","c.go":"one"},"groups":[
		{"name":"one","subject":"feat: one","body":"  "},
		{"name":"empty","subject":"feat: empty","body":null},
		{"name":"two","subject":"fix: two","body":" why \n"},
		{"name":"one","subject":"feat: duplicate","body":null}],
		"all_in_one":{"subject":"feat: all","body":null}}`
	p, err := generate(t, &fakeModel{answers: []string{answer}}, 8192, "auto", change("a.go", 1), change("b.go", 1), change("c.go", 1))
	if err != nil {
		t.Fatal(err)
	}
	why := "why"
	want := []Group{
		{Name: "one", Files: []string{"a.go", "c.go"}, Message: Message{Subject: "feat: one"}},
		{Name: "two", Files: []string{"b.go"}, Message: Message{Subject: "fix: two", Body: &why}},
	}
	if !reflect.DeepEqual(p.Groups, want) {
		t.Errorf("got %+v", p.Groups)
	}
}

func TestBodyModes(t *testing.T) {
	for mode, wantBody := range map[string]bool{"auto": false, "never": false, "always": true} {
		m := &fakeModel{}
		p, err := generate(t, m, 8192, mode, change("a.go", 10))
		if err != nil {
			t.Fatal(err)
		}
		if (p.AllInOne.Body != nil) != wantBody || (p.Groups[0].Body != nil) != wantBody {
			t.Errorf("%s: unexpected bodies: %+v", mode, p)
		}
		schema := string(m.requests[0].Schema.(json.RawMessage))
		if strings.Contains(schema, `"body"`) != (mode != "never") {
			t.Errorf("%s: schema %s", mode, schema)
		}
		if !strings.Contains(m.requests[0].System, bodyRules[mode]) {
			t.Errorf("%s: system prompt lacks the body rule", mode)
		}
	}
	// A body sent anyway is discarded when bodies are disabled.
	answer := `{"files":{"a.go":"x"},"groups":[{"name":"x","subject":"fix: a","body":"text"}],"all_in_one":{"subject":"fix: a","body":"text"}}`
	p, _ := generate(t, &fakeModel{answers: []string{answer}}, 8192, "never", change("a.go", 10))
	if p.AllInOne.Body != nil || p.Groups[0].Body != nil {
		t.Errorf("never: body kept: %+v", p)
	}
}

func TestLanguage(t *testing.T) {
	for lang, want := range map[string]string{"es": "written in Spanish,", "EN": "written in English,", "Català": `written in the language "Català",`} {
		m := &fakeModel{}
		if _, err := Generate(context.Background(), m, []gitdiff.Change{change("a.go", 1)}, Options{CtxSize: 8192, Body: "auto", Lang: lang}); err != nil {
			t.Fatal(err)
		}
		// The rule closes the request as well as the system prompt.
		if r := m.requests[0]; !strings.Contains(r.System, want) || !strings.Contains(r.User[strings.LastIndex(r.User, "\n"):], want) {
			t.Errorf("%s: language rule %q is missing", lang, want)
		}
	}
}

func TestSchema(t *testing.T) {
	raw := proposalSchema([]string{`z "quoted".go`, "a.go"}, "auto")
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("schema is not valid JSON: %v\n%s", err, raw)
	}
	// Property order is the order in which the model writes its answer.
	s := string(raw)
	order := []string{`"plan"`, `"files"`, `"z \"quoted\".go"`, `"a.go"`, `"groups"`, `"all_in_one"`}
	last := -1
	for _, key := range order {
		at := strings.Index(s, key+":")
		if at <= last {
			t.Fatalf("%s out of order in %s", key, s)
		}
		last = at
	}
}

func TestSubjectPattern(t *testing.T) {
	for subject, want := range map[string]bool{
		"feat: add login":                  true,
		"fix(api/v2): handle empty body":   true,
		"refactor(core)!: drop old flags":  true,
		"docs: añade guía de uso":          true,
		"feat:missing space":               false,
		"feature: unknown type":            false,
		"fix: say \"hi\"":                  false,
		"fix: " + strings.Repeat("x", 101): false,
		"fix(): empty scope":               false,
		"fix: two\nlines":                  false,
	} {
		if got := subjectRE.MatchString(subject); got != want {
			t.Errorf("%q: match %v, want %v", subject, got, want)
		}
	}
}

func TestPack(t *testing.T) {
	tests := []struct {
		sizes []int
		want  [][]int
	}{
		{nil, nil},
		{[]int{3, 3, 3}, [][]int{{0, 1, 2}}},
		{[]int{6, 5, 4, 1}, [][]int{{0}, {1, 2, 3}}},
		{[]int{2, 50, 2, 2}, [][]int{{0}, {1}, {2, 3}}},
	}
	for _, tt := range tests {
		if got := pack(tt.sizes, 10); !reflect.DeepEqual(got, tt.want) {
			t.Errorf("pack(%v) = %v, want %v", tt.sizes, got, tt.want)
		}
	}
}

func TestSplit(t *testing.T) {
	lines := strings.Repeat("one two three\n", 100)
	long := strings.Repeat("ñandú ", 200) // a single line with multi-byte characters
	for name, text := range map[string]string{"lines": lines, "one long line": long, "short": "tiny\n"} {
		tokens := llamatest.CountTokens(text)
		parts := split(text, tokens, 50)
		if strings.Join(parts, "") != text {
			t.Errorf("%s: parts do not add up to the text", name)
		}
		for i, part := range parts {
			if n := llamatest.CountTokens(part); n > 50 {
				t.Errorf("%s: part %d has %d tokens", name, i, n)
			}
			if !utf8.ValidString(part) {
				t.Errorf("%s: part %d splits a character", name, i)
			}
			if name == "lines" && !strings.HasSuffix(part, "\n") {
				t.Errorf("%s: part %d does not end at a line end", name, i)
			}
		}
	}
}
