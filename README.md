# generative-video-conversion

`gvc` re-imagines a video frame by frame using only self-hosted models:

```
video ─► frames ─► vision LLM (Ollama) ─► captions ─► Stable Diffusion (stable-diffusion.cpp) ─► frames ─► video
```

The output has the same frame count and frame rate as the input, and keeps the original audio.

## Setup

Requires Nix with flakes. Two dev shells are provided:

```sh
nix develop        # NVIDIA GPU (CUDA builds of ollama and stable-diffusion.cpp)
nix develop .#cpu  # CPU only, for development and tests
```

The CUDA build of stable-diffusion.cpp is compiled locally on first use, which takes a while.
A CUDA binary cache (see the [NixOS CUDA wiki page](https://wiki.nixos.org/wiki/CUDA)) avoids compiling the CUDA libraries themselves.

Download the image models the config needs into `./models`:

```sh
gvc models                      # SDXL-Turbo + fp16-fixed VAE (~7.3 GB)
gvc models --preset full-ref    # additionally IP-Adapter + CLIP-Vision (~2 GB)
```

The caption model is fetched on first use: Ollama pulls `qwen2.5vl:7b`, and llama-server downloads the GGUF from Hugging Face.

## Usage

```sh
gvc run input.mp4 -o output.mp4                       # preset "full"
gvc run input.mp4 --preset literal --style "watercolor painting"
gvc run input.mp4 -n 48                               # first 48 frames only, for quick tests
```

`gvc run` starts the caption server (`ollama serve` or `llama-server`) and `sd-server` when they are not already running, and stops them again.
The models therefore never share VRAM: the caption model is unloaded before image generation starts.

### Caption backend and speed

Captioning is usually slower than generating the image, so it's worth tuning:

- `caption.backend`: `ollama` (default; pulls models automatically) or `llama-server` (llama.cpp; more tuning knobs via `caption.llama_server_args`).
- `caption.image_size` (default 512): frames are scaled down before captioning. Vision models like Qwen2.5-VL spend about one token per 28×28 pixels, so a full 1080p frame costs about 10× more prompt work.
- `caption.parallel` (default 4): concurrent requests. This only applies with `caption.context = "none"`, because context captions depend on the previous caption. A `parallel` setting only reaches a server that gvc starts itself; an Ollama you started yourself uses its own `OLLAMA_NUM_PARALLEL`.

Measure on your own hardware:

```sh
gvc bench input.mp4 --frames 20                                    # both backends, configured image size
gvc bench input.mp4 --backend llama-server --image-size 512 --image-size 0
```

The bench prints seconds per frame (one request at a time, and parallel), an estimate for the whole video, and a sample caption for each combination so you can compare quality too.

The individual stages can also be run on their own, and every stage is cached in `work/<video>/`:

| Command | Does |
|---|---|
| `gvc probe in.mp4` | Show fps, frame count, resolution, audio |
| `gvc extract in.mp4` | Extract frames, find duplicates and scene cuts |
| `gvc caption in.mp4` | ...and caption every frame (`work/<video>/captions.jsonl`) |
| `gvc generate in.mp4` | ...and generate every frame (`work/<video>/gen/`) |
| `gvc run in.mp4` | ...and assemble the video |
| `gvc bench in.mp4` | Time caption backends and image sizes |

Interrupted runs resume where they stopped. If you change a setting, only the stages it affects run again.
For example, changing `generate.strength` keeps the captions.

## Modes

There are two independent propagation settings.

**Caption propagation** (`caption.context`, `--caption-context`) controls what the vision LLM sees:

| Value | LLM receives for frame N |
|---|---|
| `none` | frame N |
| `text` | frame N + caption N-1, and is asked to change only what differs |
| `text+image` | frame N + frame N-1 + caption N-1 |

`caption.window` passes more than one previous caption.

**Image propagation** (`generate.init`, `--init`) controls what the image model sees:

| Value | Generator receives for frame N | Sees original pixels |
|---|---|---|
| `none` | caption N, fixed seed | no |
| `chain` | caption N + generated frame N-1 as img2img start (`strength`) | no |
| `chain-ref` | caption N + generated frame N-1 as reference (`ref_strength`) | no |
| `source` | caption N + original frame N as img2img start (`strength`) | yes |

`chain` keeps composition and colours closely, but it can drift over long shots.
`chain-ref` passes the previous frame as a reference instead of a starting point. Style and subject carry over, but the composition can change more freely. Two reference methods are available:

- `ref_method = "ip-adapter"` (default): SD 1.5/SDXL models plus IP-Adapter weights.
- `ref_method = "ref-images"`: edit models such as FLUX Kontext or Qwen-Image-Edit. Point `model`/`server_args` at the model files.

**Presets** combine the two settings:

| Preset | Caption | Image |
|---|---|---|
| `literal` | none | none |
| `stable-text` | text | none |
| `chain` | none | chain |
| `full` (default) | text | chain |
| `full-ref` | text | chain-ref |
| `guided` | text | source |

### Chain resets, dedupe and stuck captions

- **Chain resets:** at a scene cut (`reset.scene_threshold`) both chains restart, with a fresh caption and a txt2img image. `reset.every_n` adds periodic restarts against drift.
- **Dedupe:** near-identical source frames (`dedupe.max_distance`) reuse the previous caption. With `init = "none"`, the image is reused too.
- **Stuck captions:** if a context caption is nearly identical to the previous one although the frame changed, it is redone without context. See `caption.stale_similarity` and `caption.stale_diff`.

## Configuration

See [gvc.toml](gvc.toml). It is picked up automatically from the current directory; use `--config` to choose another file.

### VRAM (10 GB cards)

SDXL-Turbo uses about 7 GB, and `chain-ref` with IP-Adapter adds about 2 GB.
sd-server places weights automatically. If you still run out of memory, add `"--vae-tiling"`, `"--offload-to-cpu"` or `"--type", "q8_0"` to `generate.server_args`.

## Development

```sh
nix develop .#cpu --command pytest
nix build   # package, runs the tests
```
