# Changelog

## 2026-04-10

### Gemini Native Audio transcription — lossless via explicit VAD

Transcription quality via `gemini-2.5-flash-native-audio-latest` brought from
partial (~22%) to full coverage (~97%). On a 4:46 min test voice message:

| Metric     | Before | After   | Change |
|------------|-------:|--------:|-------:|
| Words      |    112 |     530 |  4.7× |
| Characters |    649 |   3 108 |  4.8× |

**What changed in `transcribers/gemini.py`:**

- **Explicit VAD.** Disabled `automatic_activity_detection` and added explicit
  `activity_start` / `activity_end` signals around the audio stream so the
  model no longer interrupts itself mid-processing.
- **30-second segments.** Reduced `max_segment_bytes` from 2 min to 30 s — the
  optimal size for Live API completeness based on parameter sweeps.
- **2× real-time send speed.** Changed `asyncio.sleep(0.12)` to `0.25` per
  0.5-s audio chunk. The previous 4× rate caused the model to drop parts of
  longer utterances.
- **Dual transcription collection.** Now collects both `input_transcription`
  and `output_audio_transcription` from each session and returns whichever
  is more complete.
- **System instruction.** The model is explicitly asked to repeat every word
  verbatim in the original language, without skipping, summarizing, or
  translating.

**Background:** tuning was done by an automated agent (A5) sweeping through
`chunk_size × speed × segment_length × VAD` combinations. Winning config was
VAD-30s/2x with 532 words on the test audio (vs 429 for the previous
VAD-60s/2x default).
