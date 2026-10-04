# Streamlit UI

Run the voice appraisal interface from the project root:

```bash
streamlit run streamlit_app/app.py
```

The first audio analysis loads the local AudEERING Wav2Vec2 model. Later
analyses reuse Streamlit's resource cache. The manual ADV tab loads only the
smaller appraisal artifact.

Supported upload formats are WAV, FLAC, and OGG. Clear speech-only recordings
are recommended.
