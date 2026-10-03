package propose

import (
	"bytes"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/franelfers/easycommit/internal/gitdiff"
)

// subjectPattern is the Conventional Commits v1.0.0 subject line. It is both
// the grammar llama-server enforces and the check applied to the answer.
// Quotes are left out because the text output wraps messages in them.
// The length cap is a backstop well above what the prompt asks for: a
// subject cut mid-sentence by the grammar is worse than a long one.
const subjectPattern = `^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\([A-Za-z0-9._/-]+\))?!?: [^"\\\n]{1,100}$`

var subjectRE = regexp.MustCompile(subjectPattern)

// bodyPattern keeps a body to one paragraph: without line breaks or
// asterisks the model cannot write Markdown lists, headings or emphasis,
// which it does whatever the prompt says. The length cap is a backstop, like
// the one of the subject.
const bodyPattern = `^[^"\\\n*]{1,600}$`

const proposalPrompt = `You write Git commit messages that follow Conventional Commits v1.0.0.

The user sends the changes of a working tree, one section per file. A section holds the diff of the file, or a summary of it when the change is large. Files marked "content omitted" are binary or generated: judge them by their path.
Answer with:
- "plan": the logical changes you find, each named like a section title: two to five plain words separated by spaces. These are the groups the changes will be committed in. Files that depend on each other, such as a function and its callers, code and its tests or a feature and its documentation, are one logical change. Use as few groups as the changes allow and keep unrelated changes apart.
- "files": for every file, the name of its group, copied exactly from "plan".
- "groups": for every name in "plan", the message of the commit that holds the files given that name. The message must account for all of those files.
- "all_in_one": a single commit message that covers all the changes.

Every message has a subject of the form "type(scope): description". The type is one of feat, fix, docs, style, refactor, perf, test, build, ci, chore or revert. The scope is optional. The description is imperative, starts in lower case, has no final period, double quotes or emoji and is short: aim for a subject of 50 characters and never exceed 72. Messages state the change directly, without "this commit", "I", "we", "now" or "currently", and do not repeat in the description what the scope already says.
%s%s`

// closingRules end both the system prompt and the user message: small models
// drift back to English and to vague descriptions unless the request ends
// with them.
const closingRules = `
Detail: every description says what its diff adds, removes or changes and names it (the function, setting, rule or value): "retry upload on timeout", not "improve upload handling". Use the verb of the change itself, such as add, remove, rename, replace, forbid, require or limit. "update", "improve", "clarify", "tighten" and "refine" hide the change: a description that would fit many other commits is wrong. When a commit holds several changes, name the one or two that matter most.
Language: every group name, every description and every body must be written in %s, whatever the language of the changes. Only type and scope stay in English.`

// languages names the common ISO 639 codes, which the model follows far
// better than the codes themselves. Any other value is passed as written.
var languages = map[string]string{
	"en": "English", "es": "Spanish", "pt": "Portuguese", "fr": "French", "de": "German",
	"it": "Italian", "nl": "Dutch", "pl": "Polish", "ru": "Russian", "tr": "Turkish",
	"zh": "Chinese", "ja": "Japanese", "ko": "Korean",
}

// languageName resolves the lang setting to the wording used in prompts.
func languageName(lang string) string {
	if name, ok := languages[strings.ToLower(lang)]; ok {
		return name
	}
	return fmt.Sprintf("the language %q", lang)
}

var bodyRules = map[string]string{
	"auto":   `Set "body" to null unless the reason for the change is not obvious from the subject or the change breaks compatibility, fixes a security problem, migrates data, reverts a commit or spans several areas. Most messages need no body. When there is one, it explains why in one or two sentences of plain prose, without Markdown or line breaks.`,
	"always": `The body explains what changed and why in one or two sentences of plain prose, without Markdown or line breaks.`,
	"never":  `Messages have no body.`,
}

const summaryPrompt = `You summarize code changes so that commit messages can be written from your summaries.

The user sends one section per file with its diff. For every file, answer with one or two sentences of plain text, without Markdown, that say what changed and, when the diff shows it, why. Name the functions, types or settings involved. Do not describe formatting.`

const rollingPrompt = `You summarize a large change to one file, which the user sends in consecutive parts.

Each message has the summary of the parts seen so far and the next part of the diff. Answer with one new summary that covers the whole change up to this part: keep what the summary so far says and add what this part brings, merging them into at most 120 words of plain text without Markdown. Say what changed and, when the diff shows it, why. Write nothing else.`

const rollingUser = `File: %s (%s), part %d of %d.

Summary so far:
%s

Next part of the diff:
%s`

// section renders one file for a prompt. Files without content keep a single line.
func section(c gitdiff.Change, content string) string {
	if c.Omitted != "" {
		return fmt.Sprintf("### %s (%s, %s: content omitted)\n", c.Path, c.Status, c.Omitted)
	}
	return fmt.Sprintf("### %s (%s)\n%s\n", c.Path, c.Status, content)
}

// prop is a named property of a JSON object schema.
type prop struct {
	name   string
	schema any
}

// object builds the schema of a JSON object in which every property is
// required. Properties keep their order, which is the order the grammar
// makes the model write them in.
func object(props ...prop) json.RawMessage {
	var fields bytes.Buffer
	required := make([]string, len(props))
	for i, p := range props {
		if i > 0 {
			fields.WriteByte(',')
		}
		name, _ := json.Marshal(p.name)
		value, _ := json.Marshal(p.schema)
		fmt.Fprintf(&fields, "%s:%s", name, value)
		required[i] = p.name
	}
	names, _ := json.Marshal(required)
	return json.RawMessage(fmt.Sprintf(
		`{"type":"object","properties":{%s},"required":%s,"additionalProperties":false}`, &fields, names))
}

type schema = map[string]any

// messageProps are the properties of a commit message under a body mode.
func messageProps(body string) []prop {
	props := []prop{{"subject", schema{"type": "string", "pattern": subjectPattern}}}
	switch body {
	case "auto":
		props = append(props, prop{"body", schema{"type": []string{"string", "null"}, "pattern": bodyPattern}})
	case "always":
		props = append(props, prop{"body", schema{"type": "string", "pattern": bodyPattern}})
	}
	return props
}

// proposalSchema constrains the final answer. Listing every path as a
// required property makes the grammar assign each file exactly once. The
// plan comes first so that the model settles the groups before using them.
func proposalSchema(paths []string, body string) json.RawMessage {
	name := schema{"type": "string", "minLength": 1, "maxLength": 60}
	group := object(append([]prop{{"name", name}}, messageProps(body)...)...)
	list := func(items any) schema {
		return schema{"type": "array", "minItems": 1, "maxItems": len(paths), "items": items}
	}
	return object(
		prop{"plan", list(name)},
		prop{"files", perFile(paths, name)},
		prop{"groups", list(group)},
		prop{"all_in_one", object(messageProps(body)...)},
	)
}

// perFile is the schema of an object holding one value per path.
func perFile(paths []string, value any) json.RawMessage {
	props := make([]prop, len(paths))
	for i, p := range paths {
		props[i] = prop{p, value}
	}
	return object(props...)
}
