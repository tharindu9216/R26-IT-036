# Missing details and questions

The current evidence supports a substantial draft. These items govern whether its final claims are defensible. Missing evidence means not located here; send existing records if available rather than inventing records retrospectively. No response is needed to complete this dossier.

## Priority 1 — resolve before finalising claims

1. **Canonical result lineage.** Which named runs should be the final reported results: S-REFIT and CBT-V4, with calibration/latency as separate historical experiments? Provide the complete stress final-run log; executed commands/environment overrides for stress and CBT V4; run manifests containing date/commit, dataset/preprocessing hashes, upstream model/tokenizer revisions, checkpoint hashes, thresholds/weights, precision/device and metric implementation. Explain S-LATER versus S-REFIT, CBT-LATER versus CBT-V4, and both calibration-generation disagreements. Existing JSON/CSV files are already inspected; repeated copies alone will not settle lineage.
2. **Independence and evaluation corrections.** Does the supervisor accept a thesis reporting the current row-based experiments with explicit shared-text/internal-post limitations, or require a new group-aware evaluation? Stress refit/test share three raw strings; train/validation share at least 143 recovered posts. Correct `_best.pt` weight saving and the fixed-refit scheduler before future runs. A new run would require separate authorisation; none was performed here.
3. **Data ethics and permissions.** Provide dataset source/download records and exact licence/terms for Dreaddit and Therapist-QA/Annotated_data, the original papers/data documentation, any ethics approval or exemption letter, and any local collection/consent documents if local participants actually existed. Absence of local recruitment must be stated rather than inventing consent. Do not send identifiable raw participant/diary content.
4. **Model/commercial rights.** Provide the acquired base-model licence/conditions and MentalBERT access terms or permission correspondence. Its official CC-BY-NC-4.0 tag means commercial SaaS cannot be assumed cleared. Confirm whether Chapter 8 is only a proposal and who owns code/contributions. No blanket grant arises from repository MIT licensing.
5. **Individual contribution.** Provide a supervisor-approved component allocation/contribution statement, with shared/reused code and collaborative experiment responsibilities identified. Git authorship is insufficient. Confirm official individual title and approved objectives/RQs.
6. **University structure.** Provide the current official individual thesis template/rubric/module handbook or supervisor clarification resolving the nine-chapter contents versus eight-chapter body and the formatting PDF’s older four-chapter/minimum-length guidance. Supply AGENTS.md/BSC_THESIS_PROJECT_GUIDE.md if they exist elsewhere; neither was found here.

## Exact next files to provide, in order

These names describe the missing records; the suggested names are not claims that such files already exist.

| Requested file / record | Why it is needed |
| --- | --- |
| Current faculty individual thesis template `.docx`, marking rubric/module handbook `.pdf`, and supervisor formatting clarification | Resolve structure, declaration/AI disclosure and applicable page/format requirements |
| Signed/approved individual contribution allocation and approved project objectives/RQs | Establish individual versus shared work without inference from commits |
| Ethics approval or exemption PDF plus dataset permission/licence/download records | Establish permitted dataset use; distinguish original approval from local approval |
| Complete stress final training log and run configuration/manifest; CBT V4 invocation/environment manifest | Partial stress log and missing CBT batch-size override cannot establish full execution |
| Optuna SQLite database or complete trial-history JSON/CSV for both tasks | Verify actual searches, selected values and execution settings |
| Existing CBT OOF predictions CSV (all 1,618 train rows) with pseudonymous row index, fold, gold binary label and each model’s probability; equivalent stress OOF file if it exists | Recover alternative ensemble/threshold comparisons; summary scores alone are insufficient; exclude text and personal IDs |
| Generation-time result manifests for later evaluation and calibration, with checkpoint/data hashes | Reconcile differing predictions under reused filenames |
| Final-refit stress saved predictions/probabilities CSV for all 715 test rows, with pseudonymous index/label/model outputs only, if retained | Verify S-REFIT CMs and metrics without rerunning expensive inference |
| Primary PDFs or reference-manager `.bib`/`.ris` export for proposal [2] and [12], plus papers actually reviewed | Resolve full author/venue metadata and build the comparison matrix |
| Existing redacted API/browser/persistence/deployment test logs and real synthetic-input UI screenshots | Demonstrate the product rather than presenting mock screenshots as evidence |
| Model/data rights documentation, redacted invoices/usage exports and stakeholder-study records if they exist | Support commercialisation beyond proposal budgets; no such empirical evidence is assumed |

## Priority 2 — product/validation evidence gaps

The mocked transport check timed out, so API/persistence end-to-end success remains unverified. Thirteen direct SQLite assertions passed with module reload; full process restart and API-level persistence still need testing. Needed tests include core prediction and optional topic failure handling; input/limit bounds; startup missing/corrupt artifacts and offline tokenizers; HTTP/browser end-to-end operation; real current five-member XAI; concurrent prediction/explanation; explanation status and model-update provenance; browser/mobile/accessibility acceptance. Attach only actual screenshots/logs with synthetic input or authorised redacted examples. No browser screenshots were generated for this dossier.

No confidence intervals, paired comparisons, explanation faithfulness/stability evaluation, expert rationale evaluation, joint stress/distortion gold labels, external corpus validation or user/clinical study was located. These are new evidence only if stronger corresponding claims are retained; an honest prototype/benchmark thesis can discuss their absence. Do not create a participant count or study protocol after the fact.

## Concise author confirmations

- What is the official title, approved individual objective set and boundary between C3 diary and shared agent work?
- Which final artifact identities does the supervisor want reported, and is a new independence-controlled run required?
- Was any local data collected or any participant/clinician/user study performed? If yes, where are its approvals, protocol and anonymised aggregate results?
- Is commercialisation a proposed plan only, and are model/data rights cleared for any intended deployment?
- Which current faculty template/format requirements and reference style apply?
- Privately provide formal declaration fields: name/student number, degree/specialisation, department/group, supervisor/co-supervisor, submission date, required signatures and AI-use disclosure. This dossier intentionally does not reproduce those fields.

## Repository corrections to plan, not performed

Preserve the old experiments. Fix best-epoch state restoration before saving stress `_best.pt`, pass the actual refit epoch count to scheduler construction, seed Python augmentation randomness, and record exact command/config/revisions/hashes. Rename misleading `mean_val_f1` and positive-versus-macro F1 labels in any future authorised reports. Correct in-process latency figure captions, the MentalBERT Reddit+Twitter metadata claim, active-binary versus legacy type/Longformer descriptions, and directory/layout documentation. Add generation-time model provenance to stored diary decisions/explanations before asserting stable historical explanations. Standalone multi-user diary needs explicit authentication/ownership, bounded pagination, privacy/retention and error handling before production claims. These changes need a separate implementation task; no code/config/result was changed here.
