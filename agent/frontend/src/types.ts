export interface SensorWaveformData {
  times: number[];
  bvp: number[];
  eda: number[];
  bvp_unit: string;
  eda_unit: string;
}

export type SensorStatus =
  | { mode: "idle"; ready: false }
  | { mode: "skipped"; ready: true }
  | {
      mode: "sensor";
      state: "connecting" | "calibrating" | "warming_up" | "running" | "error" | "stopped";
      ready: boolean;
      calibration_remaining_seconds: number;
      calibration_total_seconds: number;
      warmup_remaining_seconds: number;
      error: string | null;
      sample_count: number;
      elapsed_seconds: number;
      serial_line_count: number;
      invalid_line_count: number;
      data_progress_age_seconds: number | null;
      stream_stalled: boolean;
      sensor_waveform?: SensorWaveformData;
      sensor_stress?: boolean;
      sensor_stress_probability?: number;
      sensor_prediction_window_end?: number;
      sensor_smoothing_count?: number;
      sensor_smoothing_window?: number;
      questionnaire_stress_available?: boolean;
      questionnaire_stress_probability?: number;
      multimodal_stress_available?: boolean;
      multimodal_stress?: boolean;
      multimodal_stress_probability?: number;
      multimodal_stress_weights?: Record<string, number>;
    };

export interface ChatReadiness {
  ready: boolean;
  state: "waiting_for_sensor" | "waiting_for_assessment" | "connecting" | "calibrating" | "warming_up" | "running" | "error" | "stopped" | "stalled";
  message: string;
  calibration_remaining_seconds: number;
  warmup_remaining_seconds: number;
}

export interface SupportContact {
  name: string;
  phone: string;
  note: string;
}

export interface QuestionnaireStatus {
  completed: boolean;
  stress?: boolean;
  stress_probability?: number;
}

export interface DASS21Form {
  version: string;
  title: string;
  timeframe: string;
  instructions: string;
  questions: { id: string; text: string }[];
  options: { value: number; label: string }[];
}

export interface DASS21Status {
  completed: boolean;
  assessment_id?: string;
  completed_at?: number;
  valid_until?: number;
}

export type DASS21Scale = "depression" | "anxiety" | "stress";
export interface DASS21Assessment {
  assessment_id: string;
  version: string;
  completed_at: number;
  answers: Record<string, number>;
  scores: Record<DASS21Scale, { raw_score: number; score: number; severity: string }>;
}

export type ModelLoadState = "pending" | "loading" | "ready" | "error";

export interface TextModelStatus {
  ready: boolean;
  models: Record<
    "stress_language" | "cognitive_patterns" | "current_emotion" | "next_turn_forecast",
    { status: ModelLoadState; error: string | null }
  >;
}

export interface VoiceModelStatus {
  ready: boolean;
  models: Record<
    "speech_recognition" | "vocal_cues" | "current_emotion" | "next_turn_forecast" | "spoken_replies",
    { status: ModelLoadState; error: string | null }
  >;
}

export interface ChatMessageMeta {
  route?: string;
  source?: string;
  current_emotion?: string;
  next_emotion?: string;
  next_emotion_confidence?: number;
  // The forecaster's own output: "low" / "high" for the next turn, and the
  // blended P(high) behind it. next_emotion is that call named as a state.
  next_intensity?: string;
  next_intensity_probability?: number;
  fusion_method?: string;
  input_text?: string;
  turn_id?: string;
  audio_url?: string;
  audio_duration?: number;
  voice_appraisal_state?: string;
  voice_stress?: boolean;
  voice_stress_score?: number;
  voice_appraisal_confidence?: number;
  voice_appraisal_uncertain?: boolean;
  previous_emotion?: string;
  deviation_level?: string;
  deviation_score?: number;
  support_contacts?: SupportContact[];
}

/** One component's contribution to a turn, as recorded by backend trace.py. */
export interface TraceOutput {
  label: string;
  value: string;
}

export interface TraceReason {
  code: string;
  text: string;
}

export interface TraceStep {
  key: string;
  component: string;
  title: string;
  /** "skipped" means the component did not run this turn, and why is in detail. */
  status: "ran" | "skipped";
  summary?: string;
  detail?: string;
  outputs?: TraceOutput[];
  /** Present on the decision steps: which conditions actually fired. */
  reasons?: TraceReason[];
}

export interface TurnTrace {
  version: number;
  turn_id?: string;
  mode: "text" | "voice";
  route: string;
  steps: TraceStep[];
}

export interface ChatMessage {
  created_at?: number;
  explanation_available?: boolean;
  role: "user" | "assistant";
  text: string;
  meta?: ChatMessageMeta;
  /** Assistant messages only, and only for turns recorded with a trace. */
  trace?: TurnTrace;
}

export interface ChatResponse {
  turn_id: string;
  reply: string;
  support_contacts: SupportContact[];
}

export interface VoiceAppraisal {
  arousal: number;
  dominance: number;
  valence: number;
  dominant_state: string;
  dominant_weight: number;
  state_strengths: Record<string, number>;
  interpretation_confidence: number;
  appraisal_ambiguity: number;
  reference_similarity: number;
  uncertain: boolean;
  ood: boolean;
  low_confidence: boolean;
  voice_stress: boolean;
  voice_stress_score: number;
  contributions: XAIAttribution[];
}

/** Whether the server can score camera frames; the UI only offers the camera when it can. */
export interface FaceModelStatus {
  available: boolean;
  error: string | null;
}

export interface VoiceChatResponse extends ChatResponse {
  transcript: string;
  duration_seconds: number;
  reply_audio_url: string;
}

export interface XAIAttribution {
  label: string;
  score: number;
}

export interface XAISection {
  target: string;
  method: string;
  member?: string;
  attributions: XAIAttribution[];
}

export interface XAICoverageItem {
  component: string;
  status: "explained" | "recorded" | "not_used" | "boundary" | "unavailable" | "output_only";
  detail: string;
}

export interface XAIResponse {
  mode?: "text" | "voice";
  sensor: XAISection | null;
  questionnaire: XAISection | null;
  combined: XAISection | null;
  stress: XAISection | null;
  cbt: XAISection | null;
  voice: XAISection | null;
  current_emotion: XAISection | null;
  next_emotion: XAISection | null;
  errors: Record<string, string>;
  note: string;
  coverage?: XAICoverageItem[];
  resource_note?: string;
}

export interface Account {
  id: string;
  name: string;
  email: string;
  role: "patient" | "doctor";
  doctor_id: string | null;
}

export interface PatientSummary extends Account {
  message_count: number;
  latest_activity: number | null;
}

export interface DoctorSummary {
  id: string;
  name: string;
  email: string;
}

export type StressSourceKey = "combined" | "questionnaire" | "sensor" | "text" | "voice" | "face";

export interface WellbeingSummary {
  days: number;
  start: string;
  end: string;
  totals: {
    check_ins: number;
    previous_check_ins: number;
    text: number;
    voice: number;
    active_days: number;
    crisis: number;
    contacts_offered: number;
    high_deviation: number;
    high_intensity_forecasts: number;
    out_of_scope: number;
  };
  negative_share: number | null;
  previous_negative_share: number | null;
  sources: {
    key: StressSourceKey;
    label: string;
    average: number | null;
    previous_average: number | null;
    change: number | null;
    readings: number;
  }[];
  daily: {
    date: string;
    check_ins: number;
    stress: Record<StressSourceKey, number | null>;
    dominant_emotion: string | null;
    crisis: number;
    contacts_offered: number;
  }[];
  emotions: { label: string; count: number; share: number }[];
  events: { time: string; mode: string; kind: "crisis" | "contacts_offered" }[];
}
