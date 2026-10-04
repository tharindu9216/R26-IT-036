# C2 voice runtime

This package is the voice-only analysis and audio-I/O branch of the supportive
agent. It contains the runtime subset migrated from the former standalone C2
research folder.

```text
browser voice recording
  -> decode once to 16 kHz mono PCM
  -> AudEERING Wav2Vec2 -> arousal/dominance/valence -> fuzzy appraisal
  -> Faster-Whisper -> transcript -> current/next emotion
  -> C1 + C2 + emotion signals -> voice LangGraph -> Qwen
  -> Kokoro -> WAV reply
```

The voice LangGraph deliberately has no C3 Stress or CBT nodes. C3 remains a
text-mode-only component. C2 appraisal output is a research conditioning
signal, not a clinical stress diagnosis.

All weights are local under `../model`: `c2/`, `fasterwhisper/`, `kokoro/`,
the current/next emotion artifacts, and Qwen/ESConv. Runtime paths and device
placement are configured in `../backend/config.yaml`.
