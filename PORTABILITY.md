# Will this run on an M1 Mac?

Short answer: **yes, and the architecture already bends the right way** — but the latency numbers
in `RESEARCH.md` are one machine's opinion and should not be quoted as portable.

Verified by checking wheel availability and reading the code for platform assumptions. **Not**
verified by running it on an M1 — nobody has, and this document says so rather than implying
otherwise.

---

## The dependency stack

| package | Apple Silicon | how |
|---|---|---|
| `ctranslate2` | ✅ | native `macosx_11_0_arm64` wheels |
| `onnxruntime` | ✅ | native `macosx_14_0_arm64` wheels |
| `piper-tts` | ✅ | `cp39-abi3-macosx_11_0_arm64` wheel |
| `faster-whisper` | ✅ | pure Python (`py3-none-any`) over ctranslate2 |
| Ollama | ✅ | native app, Metal-accelerated |
| PyMuPDF / Pillow / NiceGUI | ✅ | universal or arm64 wheels |

Nothing in the local stack is Windows-only, and nothing needs compiling from source.

## The CPU-over-CUDA decision pays off here

Choosing CPU for speech was made for local reasons — `ctranslate2` wanted `cublas64_12.dll`, and
the CUDA wheels are ~700 MB for a task that already runs in 0.6 s on CPU. On Apple Silicon that
choice stops being a convenience and becomes the only option: **there is no CUDA on a Mac at all.**
A CUDA-dependent speech path would have been dead on arrival.

Apple Silicon CPU inference is also unusually good for this workload — CTranslate2 uses the
Accelerate framework and NEON, and the unified memory means no host↔device copy. Expect the same
order of magnitude, not a cliff. Unmeasured.

## Where an M1 actually differs: memory, not architecture

Ollama on Metal is fast, but **unified memory is the constraint** — the model shares RAM with
everything else:

| model | size | 8 GB M1 | 16 GB | 32 GB+ |
|---|---|---|---|---|
| `translategemma:4b` | 3.3 GB | ✅ the default | ✅ | ✅ |
| `translategemma:12b` | 8.1 GB | ✗ | tight | ✅ |
| `gemma3:12b` | 8.2 GB | ✗ | tight | ✅ |
| `translategemma:27b` | 17.4 GB | ✗ | ✗ | ✅ |

**This is already handled.** `TranslationBackend.choose_local_model()` picks the best *installed*
model from a measured preference order, so an 8 GB M1 with only `translategemma:4b` pulled uses it
automatically — no config, no code change. And the model that wins the benchmark is also the
smallest of the serious ones (3.3 GB), which is a happy accident of task-specific models beating
big general ones.

If nothing local is reachable, the live path falls back to hosted with a 0.6 s probe cached for
60 s, so a Mac with no Ollama behaves exactly like a laptop on a plane: slower, still working.

## What would actually need attention

1. **Fonts.** `image_compositor.py` already lists `/Library/Fonts/Arial.ttf` and
   `/System/Library/Fonts/Supplemental/Arial.ttf` in its fallback chain, so overlay text should
   render. Worth an eyeball on a real Mac — a silent fallback to PIL's bitmap face is exactly the
   bug that produced `Padr▯n` and `▯8.50` on Windows.
2. **Dev commands.** Everything documented uses `.venv\Scripts\python.exe`; on macOS that is
   `.venv/bin/python`. Docs issue, not a code issue.
3. **Re-run the benchmark.** `passage/model_bench.py` appends to `data/model_bench.jsonl`, so an
   M1 run can be compared directly against the 5090 run already recorded. The *ranking* is the
   portable claim; the milliseconds are not.

## The honest summary

The design happens to be well-suited to Apple Silicon because every choice that was made for
correctness — CPU speech, runtime model selection from what's installed, hosted fallback on every
path, no CUDA anywhere — is also what makes it portable. None of that was foresight about Macs;
it fell out of refusing to hard-code the environment.

What is *not* known: real latency, whether Metal changes the model ranking, and whether the
overlay font chain resolves. All three are one afternoon on the actual hardware.
