# AI-Driven Multimodal Stress and Emotion Analysis with Explainable Intelligence

## Project Code

R26-IT-036

## Institution

Sri Lanka Institute of Information Technology (SLIIT)

## Degree Programme

B.Sc. (Hons) Degree in Information Technology

## Research Area

Artificial Intelligence, Mental Health Analytics, Multimodal Learning, Natural Language Processing, Speech Emotion Recognition, Physiological Signal Processing, Explainable AI

---

## Overview

This research project proposes an **AI-Driven Multimodal Stress and Emotion Analysis System with Explainable Intelligence**. The system is designed to detect, analyze, and explain stress and emotion-related patterns using multiple input modalities, including physiological signals, speech, text, and dialogue context.

The project combines four individual research components into one integrated system. Each component focuses on a different modality or intelligence layer. Together, these components aim to provide a more transparent, reliable, and supportive AI-based mental well-being solution.

The system does not only predict stress or emotion labels. It also provides explainable outputs, calibrated confidence scores, and supportive dialogue responses. This makes the system more suitable for sensitive domains such as digital well-being, student support, and mental health-related research.

> **Disclaimer:** This project is developed for academic research purposes only. It is not a medical diagnosis system and should not be used as a replacement for professional mental health support.

---

## Main Research Objective

The main objective of this research is to design and develop an explainable multimodal AI system that can detect stress and emotional states from physiological signals, speech, and text, and generate supportive dialogue responses using emotion forecasting and strategy-guided response generation.

---

## Specific Objectives

* Detect stress from wearable physiological sensor signals.
* Identify emotions from speech and map them into stress scores using psychological theory.
* Detect external stressors and CBT-based cognitive distortions from textual data.
* Forecast the user's next emotional state in a conversation.
* Generate supportive and safe dialogue responses based on the user's emotional condition.
* Improve model transparency using Explainable AI techniques.
* Evaluate each component separately and evaluate the complete integrated system.

---

## System Components

The proposed system contains four main components.

---

## C1: Physiological Stress Detection from Wearable Sensor Signals

Component C1 focuses on detecting stress using physiological signals collected from wearable sensors.

### Input Data

* Electrodermal Activity (EDA)
* Galvanic Skin Response (GSR)
* Electrocardiogram (ECG)
* Heart Rate
* Heart Rate Variability (HRV)

### Dataset

* WESAD: Wearable Stress and Affect Detection Dataset

### Methodology

The physiological signals are preprocessed using filtering, segmentation, and normalization. After preprocessing, important features are extracted from EDA and ECG signals. These features are used to train a lightweight hybrid CNN-LSTM deep learning model.

The CNN layers capture local signal patterns, while the LSTM layers learn time-based sequential dependencies. To improve prediction reliability, probability calibration is applied using temperature scaling. SHAP is used to explain which physiological features contributed most to the stress prediction.

### Expected Output

The expected output of C1 is a calibrated stress probability, stress label, confidence score, and feature-level explanation.

Example output:

```json
{
  "stress_probability": 0.82,
  "stress_label": "stress",
  "confidence": 0.78,
  "important_features": ["EDA_mean", "HRV", "skin_conductance_peaks"]
}
```

---

## C2: Speech Emotion Recognition and Appraisal-Based Stress Mapping

Component C2 focuses on detecting emotion from speech and mapping the detected emotion into a stress score using Appraisal Theory.

### Input Data

* Speech audio recordings
* Acoustic features extracted from voice

### Dataset

* SAVEE: Surrey Audio-Visual Expressed Emotion Dataset

### Methodology

The speech signal is converted into audio features such as log-mel spectrograms, MFCCs, pitch, and energy. These features are used to train a CNN-LSTM or Transformer-based Speech Emotion Recognition model.

After detecting the emotion, an Appraisal Theory-based mapping module converts the detected emotion and arousal information into a calibrated stress score. This helps avoid treating emotion and stress as the same concept. Grad-CAM is applied to spectrograms to explain which time-frequency regions influenced the model prediction.

### Expected Output

The expected output of C2 is a detected emotion, emotion probability distribution, stress score, stress label, and Grad-CAM-based explanation heatmap.

Example output:

```json
{
  "emotion": "anger",
  "emotion_probabilities": {
    "anger": 0.74,
    "sadness": 0.12,
    "neutral": 0.08
  },
  "stress_score": 0.81,
  "stress_label": "high_stress",
  "explanation_heatmap": "gradcam_output.png"
}
```

---

## C3: Explainable Text-Based Stressor and Cognitive Distortion Detection

Component C3 focuses on detecting external stressors and internal cognitive distortions from textual data.

### Input Data

* User-written text
* Social media-style posts
* Mental health-related text expressions

### Datasets

* Dreaddit Stress Detection Dataset
* CBT Cognitive Distortion Dataset

### Methodology

The text is preprocessed and passed into transformer-based models such as RoBERTa or DeBERTa. A multi-task learning approach is used to detect both stressors and cognitive distortions.

The stressor detection head identifies external causes of stress, such as academic pressure, financial problems, relationship issues, or workplace difficulties. The cognitive distortion detection head identifies internal thinking patterns such as catastrophizing, overgeneralization, personalization, and negative filtering.

To improve interpretability, SHAP and LIME are used to highlight important words or phrases that influenced the prediction. BERTopic is also used to discover emerging stress-related themes beyond predefined categories.

### Expected Output

The expected output of C3 is a stress detection result, stressor category, cognitive distortion label, highlighted rationale tokens, and discovered topic.

Example output:

```json
{
  "stress_detected": true,
  "stressor_category": "academic_pressure",
  "cognitive_distortion": "catastrophizing",
  "rationale_tokens": ["I will fail", "everything is ruined"],
  "discovered_topic": "exam stress and academic uncertainty"
}
```

---

## C4: Emotion Forecasting and Strategy-Guided Supportive Dialogue Generation

Component C4 focuses on forecasting the user's next emotional state and generating supportive dialogue responses.

### Input Data

* Current dialogue context
* Current emotion
* Stress score from C1 and C2
* Stressor category from C3
* Cognitive distortion label from C3
* Previous conversation turns

### Datasets

* DailyDialog
* EmpatheticDialogues
* ESConv

### Methodology

C4 follows a Forecast-then-Respond pipeline. First, the system identifies the user's current emotional state. Then it predicts the likely next-turn emotion based on the current dialogue context. After that, a support strategy is selected using the current emotion and forecasted emotion. Finally, a supportive response is generated using the selected strategy.

This approach makes the dialogue system more proactive because it does not only respond to the current emotion. It also considers how the user's emotion may change in the next turn.

### Support Strategies

* Listening
* Comforting
* Reassuring
* Encouraging
* Maintaining tone
* Safe fallback response

### Expected Output

The expected output of C4 is the current emotion, forecasted next emotion, selected support strategy, and supportive response.

Example output:

```json
{
  "current_emotion": "sadness",
  "forecasted_next_emotion": "anxiety",
  "selected_strategy": "reassuring",
  "supportive_response": "It sounds like this situation feels overwhelming. Let's take it step by step and focus on what you can control right now."
}
```

---

## Integrated System Flow

The complete system works as a multimodal pipeline.

First, physiological signals are analyzed by C1 to detect stress from wearable sensor data. Second, speech input is analyzed by C2 to detect emotion and generate an appraisal-based stress score. Third, text input is analyzed by C3 to identify stressors, cognitive distortions, and emerging stress themes. Finally, C4 integrates the outputs from the previous components and generates a supportive dialogue response.

The final system provides stress predictions, emotion predictions, explanations, and supportive responses in a single integrated framework.

---

## Key Features

* Multimodal stress and emotion analysis
* Physiological signal-based stress detection
* Speech emotion recognition
* Appraisal Theory-based stress mapping
* Text-based stressor detection
* CBT-based cognitive distortion detection
* Emotion forecasting
* Strategy-guided supportive response generation
* Explainable AI support
* Calibrated confidence scores
* Safety-aware response generation
* Modular component-based architecture

---

## Technologies Used

### Programming Language

* Python

### Machine Learning and Deep Learning

* PyTorch
* Scikit-learn
* Hugging Face Transformers
* CNN
* LSTM
* Transformer models

### Signal Processing

* NumPy
* Pandas
* SciPy
* ECG signal processing
* EDA / GSR signal processing

### Audio Processing

* Librosa
* Torchaudio
* Log-mel spectrograms
* MFCC feature extraction
* Pitch and energy extraction

### Natural Language Processing

* RoBERTa
* DeBERTa
* BERTopic
* UMAP
* HDBSCAN
* NLTK

### Explainable AI

* SHAP
* LIME
* Grad-CAM
* Token-level rationale highlighting

### Experimentation and Deployment

* Jupyter Notebook
* Google Colab
* Docker
* GitHub

---

## Evaluation Metrics

### C1 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Expected Calibration Error
* SHAP explanation quality

### C2 Evaluation Metrics

* Accuracy
* Per-class precision
* Per-class recall
* Macro F1-score
* ROC-AUC
* Expected Calibration Error
* Grad-CAM explanation analysis

### C3 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Macro-F1
* Token-level explanation validation
* Topic coherence score

### C4 Evaluation Metrics

* Current emotion classification accuracy
* Next-turn emotion forecasting accuracy
* Macro-F1
* BERTScore
* Human evaluation for empathy
* Human evaluation for relevance
* Human evaluation for helpfulness
* Human evaluation for safety
* Unsafe response rate
* Fallback trigger rate

---

## Expected Outcomes

The expected outcomes of this research project are:

* A physiological stress detection model with calibrated stress probability.
* A speech-based emotion recognition model with Appraisal Theory-based stress mapping.
* A text-based stressor and cognitive distortion detection model with token-level explanations.
* A topic discovery module for identifying emerging stress themes.
* A dialogue system that forecasts emotional trajectory and generates supportive replies.
* Explainable outputs for physiological, speech, and text-based predictions.
* A modular AI framework that can be extended for future mental well-being applications.

---

## Ethical Considerations

This project deals with sensitive mental health-related data. Therefore, ethical handling of data and responsible AI design are important.

The project follows these ethical principles:

* Use publicly available and anonymized datasets.
* Avoid storing personally identifiable information.
* Do not provide medical diagnosis.
* Do not replace professional mental health support.
* Apply safety filters for generated responses.
* Provide confidence scores and explanations.
* Treat model outputs as research-based predictions, not clinical decisions.

---

## Limitations

* Public datasets may not fully represent real-world stress behavior.
* Wearable sensor signals can vary significantly between individuals.
* SAVEE is an acted speech dataset, so it may not fully capture natural stress expressions.
* Text-based models may misinterpret sarcasm, cultural context, or indirect emotional expressions.
* Explainable AI methods provide useful interpretations but may not fully explain all internal model behavior.
* Human evaluation is required to properly assess supportive dialogue quality.

---

## Research Team

| Member              | Student ID | Component                                                   |
| ------------------- | ---------- | ----------------------------------------------------------- |
| Perera K.T.K.       | IT22235688 | C1 - Physiological Stress Detection                         |
| Chandrasena H.P.    | IT22083814 | C2 - Speech Emotion Recognition and Stress Mapping          |
| Munasinghe M.A.C.D. | IT22252586 | C3 - Text-Based Stressor and Cognitive Distortion Detection |
| Perera T.M.S.       | IT22138668 | C4 - Emotion Forecasting and Supportive Dialogue Generation |

---

## Supervisors

* Prof. Samantha Thelijjagoda
* Senior Lecturer Junius Anjana

---

## Acknowledgement

The research team acknowledges the guidance and support provided by the Department of Information Technology, Sri Lanka Institute of Information Technology, the project supervisors, and the AIMS Research Cluster.

---

## License

This repository is intended for academic research purposes. An appropriate open-source license can be added before public release.



---

## Final Disclaimer

This project is not a clinical mental health diagnosis system. The outputs generated by the models are intended only for academic research, early stress-pattern analysis, and digital well-being support. For serious mental health concerns, users should seek support from qualified mental health professionals.
