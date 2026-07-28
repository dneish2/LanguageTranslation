import time, os
SP = os.path.join(os.environ["TEMP"],"claude","C--Users-David-Coding",
                  "17c8761a-c8ba-40b4-bc3f-794fb72a9aad","scratchpad")
wav = os.path.join(SP, "speech.wav")   # real 48kHz speech from the TTS fixture
from faster_whisper import WhisperModel
for size in ("base", "small"):
    try:
        t = time.time()
        m = WhisperModel(size, device="cuda", compute_type="float16")
        load = time.time() - t
        t = time.time()
        segments, info = m.transcribe(wav, beam_size=1)
        text = " ".join(s.text for s in segments).strip()
        print(f"[{size}] load {load:4.1f}s  transcribe {time.time()-t:4.1f}s  lang={info.language}")
        print(f"      {text[:150]}")
    except Exception as e:
        print(f"[{size}] FAILED {type(e).__name__}: {str(e)[:160]}")
