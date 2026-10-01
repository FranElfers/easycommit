# easycommit

Proposes [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) messages for the changes in a Git working tree. The messages are written by a local model served by [llama.cpp](https://github.com/ggml-org/llama.cpp), so nothing leaves the machine.

easycommit only prints proposals. It never commits, stages or changes anything in the repository.

```console
$ easycommit
=== All In One =============================================

"feat(store): load key-value pairs from a file"

=== Divided in 2 commits ===================================

"feat(store): load key-value pairs from a file"
Read the file named by SAMPLE_DB at startup instead of using a fixed greeting.
├─ cmd/main.go
└─ internal/store/store.go

"docs: explain the SAMPLE_DB variable"
└─ README.md
```

"All In One" is one message for everything. "Divided" splits the changes into the commits the model suggests: each has its message, with the body under the subject when there is one, and the files to commit together. Every changed file appears in exactly one commit.

## Requirements

- Linux on x86_64 (Arch Linux and Alpine Linux are the supported distributions).
- `git`.
- `llama-server` from llama.cpp, build b7492 (December 2025) or later, with the Vulkan backend for GPU inference:

  | Distribution | Packages |
  | --- | --- |
  | Arch Linux | `pacman -S llama-cpp ggml-vulkan` |
  | Alpine Linux 3.24 | `apk add llama-server llama.cpp-vulkan` |
  | Alpine Linux edge | `apk add llama-server libggml-vulkan` |

The model (Ministral 3 8B Instruct, `Q4_K_M`, about 5 GB) is downloaded by `llama-server` into the Hugging Face cache the first time it is needed.

## Install

Arch Linux, from the AUR:

```sh
yay -S easycommit-bin
```

Any distribution, from the static binary attached to each [release](https://github.com/franelfers/easycommit/releases):

```sh
wget -O easycommit https://github.com/franelfers/easycommit/releases/latest/download/easycommit-linux-x86_64
install -Dm755 easycommit /usr/local/bin/easycommit
```

From source, with Go installed: `make build`.

## Usage

Run `easycommit` anywhere inside a repository. It reads the whole working tree: staged, unstaged and untracked files, minus whatever `.gitignore` hides.

```sh
easycommit                    # text proposals
easycommit --json             # the same as JSON, for scripts
easycommit --body never       # subjects only (auto | never | always)
easycommit --lang es          # messages in another language
easycommit --cpu              # do not use the GPU
easycommit server status      # is the background llama-server running?
easycommit server stop        # stop it and free its memory
```

`easycommit --help` lists every flag.

Proposals go to stdout. Progress, warnings and errors go to stderr, so the output can be piped safely.

On a terminal the progress is a single line that updates in place and disappears at the end: the seconds spent loading the model, a bar while the model reads the changes, and the number of tokens written while it answers. When stderr is redirected, each step is printed as a plain line instead.

The first run starts `llama-server` in the background and later runs reuse it. After `idle_timeout` seconds without use the server unloads the model from memory; the process itself stays until `easycommit server stop` or a reboot.

## Configuration

Settings are read from `~/.config/easycommit/config.toml` (or the file given with `--config`). Flags override the file. Every key is optional:

```toml
server_bin   = "llama-server"   # name in PATH, or a path
model        = "mistralai/Ministral-3-8B-Instruct-2512-GGUF:Q4_K_M"  # or a path to a .gguf file
body         = "auto"           # auto | never | always
lang         = "en"             # ISO code or language name
idle_timeout = 600              # seconds of inactivity before the model is unloaded
cpu          = false            # true: never use the GPU

# Chosen from the detected VRAM unless set here:
# ctx_size   = 16384            # context size in tokens
# gpu_layers = 24               # model layers loaded on the GPU

# Written by easycommit on the first run. Delete the line to detect again.
# vram_mb    = 16304

# Files reported by path only, like binaries. Setting it replaces the default
# list (lock files, vendor/, node_modules/, dist/, build/, minified files).
# exclude    = ["go.sum", "vendor/", "*.min.js"]
```

In `exclude`, a pattern ending in `/` matches a directory at any depth, a pattern containing `/` is a glob over the whole path, and any other pattern is a glob over the file name.

On the first run easycommit detects the VRAM of the GPU and picks a context size and a number of GPU layers for it. With little VRAM part of the model runs on the CPU. Without a usable GPU everything runs on the CPU and a warning is printed.

## Troubleshooting

- **`llama-server exited during startup`**: the message ends with the server's own log. `out of memory` means the GPU memory is taken by something else or the profile is too large: lower `ctx_size` or `gpu_layers`, or use `--cpu`. `invalid argument` means llama.cpp is older than b7492.
- **`too many changed files for the context size`**: raise `ctx_size`, or add generated files to `exclude`.
- **Messages in the wrong language**: small models drift back to English. `lang` works best with widely spoken languages.

## Documentation

- [SPECS.md](SPECS.md): the product specification (in Spanish).
- [docs/architecture.md](docs/architecture.md): how the code works.
- [docs/development.md](docs/development.md): testing, VRAM profiles and releases.

## License

[MIT](LICENSE)
