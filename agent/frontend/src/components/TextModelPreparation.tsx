import { api } from "../api";
import type { TextModelStatus } from "../types";
import { ModelPreparation } from "./ModelPreparation";

interface Props {
  onReady: () => void;
}

const MODEL_DETAILS = [
  {
    key: "stress_language",
    name: "Stress Header",
    description: "Loading the C3 BERT + DeBERTa-v3 stress ensemble.",
  },
  {
    key: "cognitive_patterns",
    name: "CBT Header",
    description: "Loading the C3 BERT + MentalBERT + DeBERTa-v3 CBT ensemble.",
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
] as const;

const INITIAL_STATUS: TextModelStatus = {
  ready: false,
  models: {
    stress_language: { status: "pending", error: null },
    cognitive_patterns: { status: "pending", error: null },
    current_emotion: { status: "pending", error: null },
    next_turn_forecast: { status: "pending", error: null },
  },
};

export function TextModelPreparation({ onReady }: Props) {
  return (
    <ModelPreparation
      details={MODEL_DETAILS}
      initialStatus={INITIAL_STATUS}
      getStatus={api.textModelStatus}
      prepare={api.prepareTextModels}
      onReady={onReady}
      eyebrow="Loading text-path models"
      title="Preparing the text analysis pipeline"
      description="Loading the Stress, CBT, current-emotion, and next-emotion forecast models."
      note="The conversation opens when all four models are ready."
      variant="text"
      fallbackError="Text conversation could not be prepared"
    />
  );
}
