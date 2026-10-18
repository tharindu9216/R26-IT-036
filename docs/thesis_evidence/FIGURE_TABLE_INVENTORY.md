# Figure and table inventory

Paths are repository relative. Assets below were inventoried by filename/size and PNG-header dimensions; existence does not authenticate the producing run. Three aggregate plots were visually inspected: CBT-V4 confusion matrices, CAL-S reliability and FUSION-HIST warm latency. Word/PDF requirements call for chapter-numbered captions and in-text references in the eventual thesis. The following does not modify any figure.

## Preferred figures and provenance restrictions

| Asset family | Experiment / suitable chapter | Usage restriction |
| --- | --- | --- |
| `reports/c3_text_stressor_distortion/CBT header/binary_v4/confusion_matrices.png` | CBT-V4 / Chapter 6 | Visually matches the six recomputed CMs and macro-F1; N506, model-specific thresholds |
| `reports/c3_text_stressor_distortion/Calibration/stress/reliability_diagram.png` and CBT analogue | CAL-S/C / Chapter 6 | Use evaluation N572/404; title “temperature scaling” needs affine-logit explanation; no classification gain |
| `reports/c3_text_stressor_distortion/Fusion/plots/2_predict_latency_distribution.png` | FUSION-HIST / Chapter 6 | Visually matches saved median/p95; plot POST title misstates in-process measurement; caption must disclose no network/browser and warmup excluded |
| Later stress/CBT evaluation plots | S-LATER / CBT-LATER / Chapters 6–7 | Do not pair with earlier refit scores; check positive-versus-macro F1 and checkpoint identity |
| BERTopic plots/HTML/topic tables | TOPIC-S/C / Chapters 3,6,7 | Training fit-subset denominators; topics are exploratory; text-bearing topics/documents need privacy review |
| XAI PNG/HTML/JSON/text examples | Historical member explanations / Chapters 4,6,7 or appendix | Not proof of current ensemble faithfulness; can contain sensitive words/source text; privacy review and exact member/config needed |
| `test/c3_text_stressor_distortion/report/` charts | DEMO / Chapter 5 or appendix | Handcrafted N10 per task; not benchmark accuracy |
| Mermaid architecture/selection diagrams in dossier | Current inspected code / Chapters 3–4 | Newly authored from code, not screenshots or external deployment evidence |

## Existing asset inventory

The inventory includes images, HTML visualisations and CSV/table outputs so that provenance is not lost. It is not a licence to reproduce raw examples. `privacy review` means text-bearing outputs may contain source material; they were not copied into the dossier. Numeric-only summary files are preferable for figures.

| Existing asset path | Bytes | Dimensions | Provenance identity | Suitable chapter | Restriction |
| --- | --- | --- | --- | --- | --- |
| `reports/c3_text_stressor_distortion/Fusion/state_distribution.csv` | 118 | not PNG / no dimension read | FUSION-HIST | 6 | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/Fusion/rule_coverage_samples.csv` | 49370 | not PNG / no dimension read | FUSION-HIST | 6 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Fusion/explain_latency_samples.csv` | 189 | not PNG / no dimension read | FUSION-HIST | 6 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/token_lengths_3models.png` | 65591 | 2684×740 | historical; resolve run | 3,6 or appendix | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/CBT header/token_lengths_3models.png` | 103534 | 3584×740 | historical; resolve run | 3,6 or appendix | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/Fusion/plots/2_predict_latency_distribution.png` | 101411 | 2669×1768 | FUSION-HIST | 6 | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/Fusion/plots/1_state_distribution.png` | 163040 | 2670×1769 | FUSION-HIST | 6 | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/Calibration/cbt/reliability_diagram.png` | 156922 | 1331×1147 | CAL-C | 6 | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/Calibration/cbt/calibration_predictions.csv` | 25818 | not PNG / no dimension read | CAL-C | 6 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Calibration/stress/reliability_diagram.png` | 152552 | 1331×1147 | CAL-S | 6 | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/Calibration/stress/calibration_predictions.csv` | 36620 | not PNG / no dimension read | CAL-S | 6 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/eda_outputs/subreddit_counts.png` | 78716 | 1476×725 | historical; resolve run | 3,6 or appendix | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/Stress header/eda_outputs/top_words_by_label.png` | 85711 | 2075×875 | historical; resolve run | 3,6 or appendix | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/Stress header/eda_outputs/simple_correlation_heatmap.png` | 227212 | 1168×1025 | historical; resolve run | 3,6 or appendix | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/Stress header/eda_outputs/class_balance.png` | 38384 | 1176×725 | historical; resolve run | 3,6 or appendix | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/Stress header/eda_outputs/subreddit_label_distribution.png` | 94329 | 1476×725 | historical; resolve run | 3,6 or appendix | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/Stress header/eda_outputs/sentiment_analysis.png` | 97343 | 1776×575 | historical; resolve run | 3,6 or appendix | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/Stress header/eda_outputs/text_length_features.png` | 119524 | 2076×1225 | historical; resolve run | 3,6 or appendix | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/Stress header/eda_outputs/top_bigrams_by_label.png` | 93183 | 2076×875 | historical; resolve run | 3,6 or appendix | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/Stress header/eda_outputs/writing_style_features.png` | 48975 | 1326×725 | historical; resolve run | 3,6 or appendix | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/Stress header/BERTopic/document_topics.csv` | 2117617 | not PNG / no dimension read | TOPIC-S | 3,6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/BERTopic/topic_by_stress_label.csv` | 8717 | not PNG / no dimension read | TOPIC-S | 3,6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/BERTopic/topic_by_subreddit.csv` | 17091 | not PNG / no dimension read | TOPIC-S | 3,6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/BERTopic/topic_terms.csv` | 18640 | not PNG / no dimension read | TOPIC-S | 3,6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/BERTopic/topic_info.csv` | 125026 | not PNG / no dimension read | TOPIC-S | 3,6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/BERTopic/plots/topic_hierarchy.html` | 3645647 | not PNG / no dimension read | TOPIC-S | 3,6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/BERTopic/plots/topic_barchart.html` | 3652053 | not PNG / no dimension read | TOPIC-S | 3,6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/BERTopic/plots/topic_sizes.png` | 268981 | 2375×1378 | TOPIC-S | 3,6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/BERTopic/plots/topics.html` | 3679562 | not PNG / no dimension read | TOPIC-S | 3,6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/BERTopic/plots/topic_stress_distribution.png` | 308335 | 1771×3687 | TOPIC-S | 3,6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/BERTopic/plots/topic_heatmap.html` | 3650160 | not PNG / no dimension read | TOPIC-S | 3,6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/error_analysis/error_summary.csv` | 993 | not PNG / no dimension read | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/error_analysis/subreddit_error_summary.csv` | 214 | not PNG / no dimension read | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/error_analysis/agreement_analysis.csv` | 1510130 | not PNG / no dimension read | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/error_analysis/combined_predictions.csv` | 1500491 | not PNG / no dimension read | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/metrics/model_comparison.csv` | 715 | not PNG / no dimension read | S-LATER | 6,7 | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/metrics/subreddit_classification_metrics.csv` | 166 | not PNG / no dimension read | S-LATER | 6,7 | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/error_analysis/plots/2_fp_fn_comparison.png` | 145933 | 3553×1753 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/error_analysis/plots/5_error_agreement_matrix.png` | 162188 | 2202×1753 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/error_analysis/plots/3_confusion_matrices.png` | 246069 | 4904×2945 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/error_analysis/plots/4_confidence_distributions.png` | 785552 | 3553×5954 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/error_analysis/plots/1_error_rate_comparison.png` | 161361 | 3553×1754 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/ig_plots/ig_sample.png` | 52265 | 1184×585 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/shap_plots/shap_sample.html` | 53928 | not PNG / no dimension read | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/shap_plots/shap_sample.png` | 46753 | 1184×584 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779854346_shap.png` | 78616 | 2234×584 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1785664390_ig.png` | 73035 | 1881×585 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1784172731_ig.png` | 54766 | 1184×585 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1784981036_integrated_hessians.png` | 100589 | 1901×2007 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779854568_shap.png` | 78616 | 2234×584 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/mentalbert_1779863965_shap.png` | 75653 | 2234×584 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1784170784_ig.png` | 73035 | 1881×585 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/bert_1779863935_ig.png` | 71488 | 1881×584 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/bert_1779863935_shap.png` | 71753 | 2234×584 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1785665370_shap.html` | 112695 | not PNG / no dimension read | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779807445_ig.png` | 72921 | 1881×585 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779854346_shap.html` | 112232 | not PNG / no dimension read | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1783828287_shap.png` | 77390 | 2234×584 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779854452_shap.html` | 112232 | not PNG / no dimension read | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1784981170_integrated_hessians.png` | 100589 | 1901×2007 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1784172348_shap.html` | 112695 | not PNG / no dimension read | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/bert_1779864150_shap.png` | 71753 | 2234×584 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/bert_1779864150_ig.png` | 71488 | 1881×584 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/mentalbert_1779864191_shap.png` | 75653 | 2234×584 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1783828287_shap.html` | 112695 | not PNG / no dimension read | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1785665370_shap.png` | 77390 | 2234×584 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/mentalbert_1779863965_ig.png` | 71442 | 1881×584 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779854452_shap.png` | 78616 | 2234×584 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/mentalbert_1779864335_shap.html` | 112030 | not PNG / no dimension read | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1785666169_combined_xai.png` | 64584 | 1584×984 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/bert_1779864150_shap.html` | 112311 | not PNG / no dimension read | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779854327_ig.png` | 72921 | 1881×585 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779855749_shap.html` | 112232 | not PNG / no dimension read | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779862660_shap.html` | 112232 | not PNG / no dimension read | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1785665429_integrated_hessians.png` | 64684 | 1019×1024 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779854692_shap.html` | 112232 | not PNG / no dimension read | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1783828215_ig.png` | 73035 | 1881×585 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1785665314_lime.png` | 60999 | 1184×928 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779856673_attention_rollout.png` | 63321 | 1881×585 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/mentalbert_1779863965_shap.html` | 112030 | not PNG / no dimension read | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/mentalbert_1779864335_shap.png` | 75653 | 2234×584 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779854692_shap.png` | 78616 | 2234×584 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/bert_1779863993_ig.png` | 71488 | 1881×584 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1784172731_shap.png` | 55099 | 1184×584 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779854162_shap.png` | 78616 | 2234×584 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/mentalbert_1779864191_shap.html` | 112030 | not PNG / no dimension read | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1785665370_ig.png` | 73035 | 1881×585 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1784172348_shap.png` | 77390 | 2234×584 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779855735_ig.png` | 72921 | 1881×585 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779854568_shap.html` | 112232 | not PNG / no dimension read | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1785664390_shap.png` | 77390 | 2234×584 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/mentalbert_1779864738_shap.png` | 75653 | 2234×584 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/mentalbert_1779864191_ig.png` | 71442 | 1881×584 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1785665314_lime.html` | 1232866 | not PNG / no dimension read | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/bert_1779863614_shap.png` | 71753 | 2234×584 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1785686173_combined_xai.png` | 64584 | 1584×984 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779854535_ig.png` | 72921 | 1881×585 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779854112_ig.png` | 72921 | 1881×585 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779862660_ig.png` | 72921 | 1881×585 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/bert_1779863614_ig.png` | 71488 | 1881×584 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/bert_1779863993_shap.html` | 112311 | not PNG / no dimension read | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779854162_shap.html` | 112232 | not PNG / no dimension read | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779862660_shap.png` | 78616 | 2234×584 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/mentalbert_1779864738_ig.png` | 71442 | 1881×584 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1783734346_ig.png` | 72921 | 1881×585 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/bert_1779863614_shap.html` | 112311 | not PNG / no dimension read | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/bert_1779863935_shap.html` | 112311 | not PNG / no dimension read | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1785665370_lime.png` | 60999 | 1184×928 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779854395_ig.png` | 72921 | 1881×585 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1783828287_ig.png` | 73035 | 1881×585 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1784172322_ig.png` | 73035 | 1881×585 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1784172348_ig.png` | 73035 | 1881×585 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779807479_shap.html` | 112232 | not PNG / no dimension read | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1785665370_lime.html` | 1232866 | not PNG / no dimension read | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1784172731_shap.html` | 49500 | not PNG / no dimension read | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779855749_shap.png` | 78616 | 2234×584 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1785665765_ig.png` | 73035 | 1881×585 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/bert_1779863993_shap.png` | 71753 | 2234×584 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1785664390_shap.html` | 112695 | not PNG / no dimension read | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779854211_ig.png` | 72921 | 1881×585 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1785686760_combined_xai.png` | 64584 | 1584×984 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779807479_shap.png` | 78616 | 2234×584 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/mentalbert_1779864738_shap.html` | 112030 | not PNG / no dimension read | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1784172720_ig.png` | 54766 | 1184×585 | S-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/metrics/plots/confusion_matrices.png` | 107825 | 2683×1535 | S-LATER | 6,7 | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/metrics/plots/2_confusion_matrices.png` | 338333 | 4438×2414 | S-LATER | 6,7 | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/metrics/plots/3_roc_curves.png` | 291862 | 2964×2364 | S-LATER | 6,7 | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/metrics/plots/6_metrics_heatmap.png` | 206449 | 2774×1764 | S-LATER | 6,7 | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/metrics/plots/4_f1_accuracy_comparison.png` | 136881 | 4169×1943 | S-LATER | 6,7 | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/metrics/plots/training_curves.png` | 213149 | 2084×1769 | S-LATER | 6,7 | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/metrics/plots/5_precision_recall.png` | 123691 | 2964×2364 | S-LATER | 6,7 | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/metrics/plots/1_metrics_comparison.png` | 210530 | 4164×1763 | S-LATER | 6,7 | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/metrics/plots/comparison_dashboard.png` | 89564 | 2977×742 | S-LATER | 6,7 | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/CBT header/binary_v4/binary_test_predictions.csv` | 65437 | not PNG / no dimension read | CBT-V4 | 6 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/CBT header/binary_v4/confusion_matrices.png` | 104388 | 3133×1842 | CBT-V4 | 6 | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/CBT header/binary_v4/training_curves.png` | 203199 | 2084×1769 | CBT-V4 | 6 | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/CBT header/binary_v4/comparison_dashboard.png` | 90355 | 2973×742 | CBT-V4 | 6 | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/CBT header/eda_outputs/top_words_by_distortion.png` | 100785 | 2975×884 | historical; resolve run | 3,6 or appendix | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/CBT header/eda_outputs/cbt_linguistic_features.png` | 250039 | 2376×1517 | historical; resolve run | 3,6 or appendix | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/CBT header/eda_outputs/top_bigrams_by_distortion.png` | 109175 | 2675×875 | historical; resolve run | 3,6 or appendix | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/CBT header/eda_outputs/class_balance.png` | 162010 | 2223×725 | historical; resolve run | 3,6 or appendix | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/CBT header/eda_outputs/distorted_span_analysis.png` | 107188 | 2076×725 | historical; resolve run | 3,6 or appendix | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/CBT header/eda_outputs/distortion_cooccurrence.png` | 154167 | 1633×1025 | historical; resolve run | 3,6 or appendix | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/CBT header/eda_outputs/text_length_features.png` | 262610 | 2372×1517 | historical; resolve run | 3,6 or appendix | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/CBT header/eda_outputs/writing_style_features.png` | 120079 | 1776×725 | historical; resolve run | 3,6 or appendix | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/CBT header/eda_outputs/correlation_heatmap.png` | 424134 | 1671×1476 | historical; resolve run | 3,6 or appendix | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/CBT header/BERTopic/document_topics.csv` | 3960297 | not PNG / no dimension read | TOPIC-C | 3,6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/CBT header/BERTopic/topic_by_distortion_status.csv` | 2667 | not PNG / no dimension read | TOPIC-C | 3,6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/CBT header/BERTopic/topic_by_distortion_type.csv` | 14298 | not PNG / no dimension read | TOPIC-C | 3,6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/CBT header/BERTopic/topic_terms.csv` | 6002 | not PNG / no dimension read | TOPIC-C | 3,6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/CBT header/BERTopic/topic_info.csv` | 33196 | not PNG / no dimension read | TOPIC-C | 3,6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/CBT header/BERTopic/plots/topic_distortion_type_distribution.png` | 317874 | 2996×1270 | TOPIC-C | 3,6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/CBT header/BERTopic/plots/topic_hierarchy.html` | 3644899 | not PNG / no dimension read | TOPIC-C | 3,6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/CBT header/BERTopic/plots/topic_barchart.html` | 3650594 | not PNG / no dimension read | TOPIC-C | 3,6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/CBT header/BERTopic/plots/topic_sizes.png` | 213481 | 2379×1338 | TOPIC-C | 3,6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/CBT header/BERTopic/plots/topics.html` | 3648941 | not PNG / no dimension read | TOPIC-C | 3,6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/CBT header/BERTopic/plots/topic_distortion_status_distribution.png` | 116468 | 1650×1270 | TOPIC-C | 3,6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/CBT header/BERTopic/plots/topic_heatmap.html` | 3647639 | not PNG / no dimension read | TOPIC-C | 3,6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/error_analysis/error_summary.csv` | 991 | not PNG / no dimension read | CBT-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/error_analysis/distortion_type_error_summary.csv` | 2484 | not PNG / no dimension read | CBT-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/error_analysis/agreement_analysis.csv` | 2381991 | not PNG / no dimension read | CBT-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/error_analysis/combined_predictions.csv` | 2375165 | not PNG / no dimension read | CBT-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/metrics/model_comparison.csv` | 713 | not PNG / no dimension read | CBT-LATER | 6,7 | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/error_analysis/plots/2_fp_fn_comparison.png` | 132441 | 3570×1769 | CBT-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/error_analysis/plots/5_error_agreement_matrix.png` | 148779 | 2208×1770 | CBT-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/error_analysis/plots/3_confusion_matrices.png` | 232194 | 4920×2952 | CBT-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/error_analysis/plots/6_error_rate_by_distortion_type.png` | 281260 | 4160×2069 | CBT-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/error_analysis/plots/4_confidence_distributions.png` | 849790 | 3570×5966 | CBT-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/error_analysis/plots/1_error_rate_comparison.png` | 146246 | 3570×1769 | CBT-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/xai_outputs/deberta_v3/deberta_v3_1785686651_ig.png` | 63761 | 1551×584 | CBT-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/xai_outputs/bert/bert_1785677214_ig.png` | 53970 | 1221×584 | CBT-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/xai_outputs/streamlit/deberta_v3_1785686799_combined_xai.png` | 65178 | 1584×984 | CBT-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/xai_outputs/streamlit/deberta_v3_1785678441_ig.png` | 63739 | 1551×584 | CBT-LATER | 6,7 | privacy review; no raw content reproduced |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/metrics/plots/2_confusion_matrices.png` | 326119 | 4435×2415 | CBT-LATER | 6,7 | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/metrics/plots/3_roc_curves.png` | 281705 | 2964×2364 | CBT-LATER | 6,7 | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/metrics/plots/6_metrics_heatmap.png` | 205042 | 2774×1764 | CBT-LATER | 6,7 | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/metrics/plots/4_f1_accuracy_comparison.png` | 137805 | 4169×1943 | CBT-LATER | 6,7 | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/metrics/plots/5_precision_recall.png` | 126860 | 2964×2364 | CBT-LATER | 6,7 | aggregate candidate; confirm run/caption |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/metrics/plots/1_metrics_comparison.png` | 206060 | 4164×1763 | CBT-LATER | 6,7 | aggregate candidate; confirm run/caption |
| `test/c3_text_stressor_distortion/report/predictions.csv` | 3179 | not PNG / no dimension read | DEMO | 5,appendix | privacy review; no raw content reproduced |
| `test/c3_text_stressor_distortion/report/positive_probability_graph.png` | 102177 | 1600×1400 | DEMO | 5,appendix | aggregate candidate; confirm run/caption |
| `test/c3_text_stressor_distortion/report/accuracy_chart.png` | 42461 | 1400×900 | DEMO | 5,appendix | aggregate candidate; confirm run/caption |
| `test/c3_text_stressor_distortion/report/confusion_matrix_chart.png` | 37989 | 1800×800 | DEMO | 5,appendix | aggregate candidate; confirm run/caption |
| `test/c3_text_stressor_distortion/report/ensemble/stress_ensemble_chart.png` | 120096 | 2780×1185 | DEMO | 5,appendix | aggregate candidate; confirm run/caption |
| `test/c3_text_stressor_distortion/report/ensemble/cbt_ensemble_chart.png` | 143448 | 2780×1185 | DEMO | 5,appendix | aggregate candidate; confirm run/caption |

## Tables prepared in the dossier

| Table | Source / identity | Chapter |
| --- | --- | --- |
| Scope/maturity and source ledger | README, research/runtime/agent code | 1,4 |
| Twelve-reference corrections and method additions | Proposal p23, primary links | 2,References |
| Processed binary distribution and eleven-class CBT distribution | Six processed CSVs and metadata; DATA-AUDIT | 3 |
| Exact partition overlap and recovered grouping explanation | Six CSVs, raw Dreaddit keys; OVERLAP/GROUP-AUDIT | 3,7 |
| Actual saved hyperparameters/refit budgets | S-REFIT JSON, CBT-V4 JSON/config | 3 |
| Experiment registry and metric definitions | Identifiable reports/code | 3,6 |
| Requirements/API/persistence evidence | Current B/F/IA code | 4,5 |
| Calibration parameters and four-state rule | Model calibration JSON, decision.py | 4 |
| Existing versus current versus needed test evidence | Test source, checks in log | 5 |
| S-REFIT historical comparison | Stress all_results.json N715 | 6 |
| S-BEST saved diagnostic CMs/metrics | Stress JSON best_fold_test arrays N715, recomputed | 6 |
| CBT-V4 CMs/thresholds/metrics | Saved test predictions N506, recomputed | 6 |
| S-LATER and CBT-LATER separate comparisons | Later prediction CSVs, recomputed | 6,7 |
| Retrospective calibration probability metrics | CAL-S/C evaluation N572/404 | 6 |
| Topic fit-subset metrics and latency narrative | TOPIC-S/C and FUSION-HIST | 6,7 |
| Chapter readiness matrix | All identified evidence and gaps | Planning |
| Artifact byte-size/SHA-256 manifest | Present273 files hashed | Appendix |

## Missing visuals and files

No actual C3 browser screenshots were located in the inspected sources. Obtain real synthetic-input captures of entry creation, four-state display, timeline, insights, explanation with partial/rule-only states and error handling; include build/commit/date. A screenshot does not establish usability or clinical validation. Need an approved contribution/scope diagram, stakeholder-approved use cases, privacy/control design and deployment topology if these are claimed as implemented. Final-refit stress CM figures need retained matching final predictions or a separately authorised reproduction; do not reuse `_best.pt`/later figures for S-REFIT. Human evaluation/stability/uncertainty figures do not exist in the located evidence and must not be fabricated. Existing topic/XAI examples require privacy and reproduction-rights review before inclusion.
