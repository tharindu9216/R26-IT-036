import { api } from "../api";
import type { VoiceModelStatus } from "../types";
import { ModelPreparation } from "./ModelPreparation";

interface Props {
  onReady: () => void;
}

const MODEL_DETAILS = [
  {
    key: "speech_recognition",
    name: "Faster-Whisper STT",
    description: "Loading the speech-to-text transcription model.",
  },
  {
    key: "vocal_cues",
    name: "AudEERING Voice Appraisal",
    description: "Loading the wav2vec2-large voice appraisal and stress model.",
  },
  {
    key: "current_emotion",
    name: "Current Emotion Header",
    description: "Loading the RoBERTa current-emotion classifier.",
  },
  {
    key: "next_turn_forecast",
    name: "Next Emotion Forecast Header",
    description: "Loading the hybrid next-turn intensity forecaster.",
  },
  {
    key: "spoken_replies",
    name: "Kokoro TTS",
    description: "Loading the text-to-speech model for spoken replies.",
  },
] as const;

const INITIAL_STATUS: VoiceModelStatus = {
  ready: false,
  models: {
    speech_recognition: { status: "pending", error: null },
    vocal_cues: { status: "pending", error: null },
    current_emotion: { status: "pending", error: null },
    next_turn_forecast: { status: "pending", error: null },
    spoken_replies: { status: "pending", error: null },
  },
};

export function VoiceModelPreparation({ onReady }: Props) {
  return (
    <ModelPreparation
      details={MODEL_DETAILS}
      initialStatus={INITIAL_STATUS}
      getStatus={api.voiceModelStatus}
      prepare={api.prepareVoiceModels}
      onReady={onReady}
      eyebrow="Loading voice-path models"
      title="Preparing the voice analysis pipeline"
      description="Loading speech recognition, voice appraisal, emotion forecasting, and spoken-reply models."
      note="The voice conversation opens when all five models are ready."
      variant="voice"
      fallbackError="Voice conversation could not be prepared"
    />
  );
}
