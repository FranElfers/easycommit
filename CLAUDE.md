# easycommit

Go CLI that proposes Conventional Commits messages for a Git working tree, using a local model served by llama.cpp.

Read before changing behaviour:

- `README.md`: what the program does for its users.
- `docs/architecture.md`: how the code does it.
- `docs/development.md`: testing approach, VRAM profiles, releasing.

## Commands

```sh
make check    # gofmt + go vet
make test     # all tests, with the race detector
make build    # static binary ./easycommit
go test ./internal/propose/ -run TestRollingSummary   # a single test
```

## Rules

- stdout carries only proposals. Progress, warnings and errors go to stderr.
- Never write to the user's repository or index, and never commit or stage for them.
- Only talk to the `llama-server` that easycommit started. Never reuse or signal another instance.
- Linux only, no CGO. The only dependency outside the standard library is `BurntSushi/toml`; ask before adding another.
- Tests must not need a GPU, a model or the network. Use `internal/llamatest`. `TestRealServer` is the only exception and is opt-in.
- Prompts and schemas live in `internal/propose/prompt.go`. The fake server cannot judge them: verify changes against the real model, as described in `docs/development.md`.
- A new setting needs a field in `config.Config`, a flag in `cli.parseArgs` and a line in the README example.
- Code and documentation are in English.
- Each fact lives in one document: the README says how to use the program, `architecture.md` how it works, `development.md` how to test and release. Link instead of repeating.
