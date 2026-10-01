# Architecture

easycommit is a short-lived command that talks to a long-lived `llama-server` it starts itself. [SPECS.md](../SPECS.md) says what the program must do; this document says how the code does it.

## Packages

| Package | Responsibility |
| --- | --- |
| `main` | Calls `cli.Run` with a context cancelled by Ctrl-C. |
| `internal/cli` | Flags, subcommands, wiring of the other packages, text and JSON output. |
| `internal/config` | Defaults, TOML file, validation, VRAM profiles. |
| `internal/gitdiff` | Reads the changes of the working tree. |
| `internal/server` | Lifecycle of the background `llama-server`, model lookup, VRAM detection. |
| `internal/llama` | HTTP client for `llama-server` over a Unix socket. |
| `internal/propose` | Prompts, token budget and the inference flows that produce proposals. |
| `internal/llamatest` | Fake `llama-server` used by the tests. |

Dependencies point one way: `cli` uses everything, `server` uses `config` and `llama`, `propose` uses `gitdiff` and `llama`, and those three use nothing internal.

## A run, step by step

1. **Options** (`cli.parse`). Flags are parsed twice: once to find `--config`, then again on top of the values loaded from that file. This gives the precedence defaults < file < flags without tracking which flags were set.
2. **Changes** (`gitdiff.Collect`). No changes ends the run here, before anything is started.
3. **Profile**. If `vram_mb` is unset, `server.DetectVRAM` measures it and `config.SaveVRAM` appends it to the file. `Config.WithProfile` then fills `ctx_size` and `gpu_layers` unless the user set them.
4. **Server** (`Server.Ensure`). Returns a client for a ready server.
5. **Proposals** (`propose.Generate`).
6. **Output**. Text or JSON to stdout.

## Reading the working tree

`gitdiff.Collect` produces one `Change` per file, with its path, status and diff.

- It copies the repository index to a temporary file and points `GIT_INDEX_FILE` at the copy. `git add --intent-to-add` on that copy makes untracked files visible to `git diff`, so a single diff against `HEAD` covers staged, unstaged and untracked changes. The real index and the object database are never written.
- A repository without commits is compared with the empty tree.
- Rename detection is off: a rename shows up as a deleted and an added file, so that every path is listed and can be staged by itself. The content of deleted files is left out because the path says enough.
- The full diff is cut into files at the `diff --git` lines and matched by position with the output of `--name-status -z`, which carries the raw paths.
- Binary files and files matching an `exclude` pattern keep only path and status (`Change.Omitted`).

## Inference

Token counts come from the server's own tokenizer (`/tokenize`). The context must hold the system prompt, the user content and the answer. The size of the answer is estimated from the number of files, since the answer names each of them, and that estimate is also the generation limit of the request.

`propose.Generate` picks one of two flows:

- **Single inference.** The diffs of all files fit: one request reads them and writes the proposal.
- **Summaries first.** They do not fit: each diff is replaced by a summary and the final request reads the summaries.
  - Files are packed in order into *blocks* that fit the context. One request per block returns a summary per file.
  - A file too large for a block gets a *rolling summary*: the diff is cut into parts at line ends, and each request receives the summary so far plus the next part and returns the updated summary.
  - Files without content need no summary.
  - If even the summaries do not fit, the run fails with `ErrTooLarge`.

Blocks are internal. Only *groups*, which come from the final request, are printed.

### Answer format

Answers are constrained with a JSON schema, which `llama-server` turns into a grammar: the model cannot produce anything else. The final answer has, in this order:

1. `plan`: the names of the groups. Writing them first makes the model settle the grouping before using it.
2. `files`: an object with one required property per path, whose value is a group name. The grammar therefore assigns every file exactly once.
3. `groups`: name, subject and body for each group.
4. `all_in_one`: subject and body.

Subjects must match the Conventional Commits pattern in `subjectPattern`, which is part of the schema. The body is nullable (`auto`), a non-empty string (`always`) or absent (`never`).

The grammar cannot check that a name used in `files` exists in `groups`, so `propose` does. An answer that fails this check, fails the subject pattern or is cut by the token limit is asked again with the reason appended to the prompt, for at most three attempts. Groups left without files are dropped.

Block summaries use the same per-path object, with a string per file.

## Server lifecycle

State lives in `$XDG_RUNTIME_DIR/easycommit/`, or `~/.cache/easycommit/run/` when that variable is unset:

| File | Purpose |
| --- | --- |
| `server.sock` | Unix socket the server listens on. |
| `server.pid` | PID of the server. |
| `server.lock` | `flock` taken while starting or stopping, so concurrent runs share one server. |
| `server.log` | Output of the server, shown when it fails to start. |

- **Identity.** A PID counts as the server only if `/proc/<pid>/cmdline` contains the socket path. This rules out recycled PIDs and any `llama-server` that easycommit did not start. Since nothing else listens on that socket, a foreign instance is never reused.
- **Settings.** The command line built by `Server.args` holds every setting that affects the server. If the running server's command line does not end with the wanted one, the server is restarted.
- **Start.** The process gets its own session (`Setsid`), which keeps it alive after easycommit and the terminal exit. `Ensure` then polls `/health` until the model is loaded and fails with the tail of the log if the process dies.
- **Idle.** `--sleep-idle-seconds` makes the server unload the model by itself. The next request loads it again. `/health` and `/props` do not wake it up, so `server status` can report the state without loading the model.
- **Stop.** `SIGTERM` to the process group, `SIGKILL` after ten seconds, then the PID file and socket are removed.

### Model

`model` is a path to a `.gguf` file or a Hugging Face repository. For a repository, `cachedModel` looks for the file in the Hugging Face cache and passes it with `--model`, which needs no network. If it is missing, the repository is passed with `--hf-repo`, `llama-server` downloads it, and its log is relayed to stderr as progress.

### VRAM

`llama-server --list-devices` prints the devices its backend can use with their memory. `DetectVRAM` takes the largest total. No devices means 0, and the server is then started with `--device none`.

## Open points of the specification

Section 8 of SPECS.md lists questions that were open when the spec was written. This is how the code settles them.

| Point | Decision |
| --- | --- |
| VRAM profiles | Tiers in `config/profile.go`, derived from measurements on one 16 GB card. Provisional: see [development.md](development.md#vram-profiles). |
| Unix socket support | Supported since llama.cpp PR #12613, which is older than the minimum version below. The TCP fallback of the spec is therefore not implemented. |
| Download cache | `llama-server` stores downloads in the Hugging Face cache, where easycommit looks. Verified with `--cache-list`. |
| VRAM detection | `llama-server --list-devices`: it reports what the inference backend will actually see and adds no dependency. With several GPUs the largest one sets the profile. Integrated graphics count with the shared memory the backend reports. |
| Minimum llama.cpp | b7492 (December 2025), the first build with `--sleep-idle-seconds`. Alpine 3.24 ships b9564 and Arch a later one. |

Two details differ from the letter of the spec:

- Alpine edge renamed the Vulkan backend package to `libggml-vulkan`. The spec's `llama.cpp-vulkan` is right for Alpine 3.24.
- `--version` and the `EASYCOMMIT_DEBUG` variable exist although the spec does not list them.
