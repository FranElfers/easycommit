# Development

The `Makefile` holds the commands: `make check` (gofmt and vet), `make test` and `make build`.

## Testing

No test needs a GPU, a model or the network. They need `git` and Linux (`/proc`).

- **Fake server.** `internal/llamatest` implements the `llama-server` command line and HTTP API with deterministic answers: one token per word, and a proposal that puts every file in a single group. `llamatest.Install` writes a `llama-server` script that re-runs the test binary as that fake, which is why packages using it call `llamatest.Main` from `TestMain`. The `server` and `cli` tests start it as a real background process, so they exercise the true lifecycle: detaching, PID file, socket, lock and restart.
- **Scripted model.** `propose` tests use an in-process `fakeModel` that can return invalid or cut answers, to cover validation and retries.
- **Isolation.** Tests point `XDG_RUNTIME_DIR`, `HF_HUB_CACHE` and `--config` at temporary directories. The runtime directory comes from `llamatest.ShortDir` because a Unix socket path is limited to 107 bytes, which `t.TempDir` can exceed.

The fake server cannot judge the quality of a prompt. After changing a prompt or a schema, check it against the real model:

```sh
# Whole program against a real llama-server and the default model (opt-in test).
EASYCOMMIT_TEST_SERVER_BIN=$(command -v llama-server) go test -run TestRealServer -v ./internal/cli/

# Dump every prompt and model answer of a run to stderr.
EASYCOMMIT_DEBUG=1 easycommit
```

Forcing a small context (`--ctx-size 4096`) on a large diff is the quickest way to exercise the block and rolling summaries.

## VRAM profiles

`config/profile.go` maps the detected VRAM to a context size and a number of GPU layers. All profiles use the same model.

| VRAM (MiB) at least | Context | GPU layers |
| --- | --- | --- |
| 14000 | 32768 | all |
| 10000 | 16384 | all |
| 7500 | 8192 | all |
| 5500 | 8192 | 24 |
| 3500 | 8192 | 12 |
| less, or no GPU | 8192 | 0 |

The tiers leave room for a desktop session next to the model. They are provisional: they come from one machine and have not been tried on smaller cards.

Measurements behind them, taken on a Radeon RX 9060 XT (16 GB) with llama.cpp b11200 and Ministral 3 8B `Q4_K_M` (34 blocks, 35 layers that can be offloaded), as the growth of used VRAM once the model is loaded:

| Context | GPU layers | VRAM used (MiB) |
| --- | --- | --- |
| 8192 | all | 6082 |
| 16384 | all | 7147 |
| 32768 | all | 9361 |
| 8192 | 24 | 4348 |
| 8192 | 12 | 2477 |

That is about 135 MiB per 1000 tokens of context and 155 MiB per layer. The build measured used the ROCm backend, so Vulkan figures may differ slightly.

To calibrate another card, run `easycommit --ctx-size N --gpu-layers M` and read the memory in use while the model is loaded (`mem_info_vram_used` under `/sys/class/drm/card*/device/` on AMD, `nvidia-smi` on NVIDIA).

## Releasing

1. Tag the commit: `git tag v0.2.0 && git push origin v0.2.0`.
2. The `Release` workflow tests, builds the static binary with the version from the tag and attaches `easycommit-linux-x86_64` and its checksum to a GitHub Release.
3. Update `packaging/aur/PKGBUILD`: set `pkgver`, run `updpkgsums`, regenerate `.SRCINFO` with `makepkg --printsrcinfo` and push both to the AUR repository of `easycommit-bin`.

The workflows have not run yet: this repository had no remote when they were written. The build step and the package were checked locally.
