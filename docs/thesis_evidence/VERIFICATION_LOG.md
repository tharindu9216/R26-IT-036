# Verification log

Audit date: 9 October 2026, Asia/Colombo. Repository HEAD `98e1d32` (4 October 2026); initial `git status --short` was empty. Inspected repository files were read only. This directory is the sole repository write target. Scratch outputs/scripts/build/database experiments are under `/tmp`; no private diary database was opened. No credentials or personal identifier values were emitted to the deliverables. No training, real model inference, paid service, public server, commit or push was run.

## Environment observed now

| Item | Observed value / limitation |
| --- | --- |
| OS/architecture | Linux 7.0.0-38, x86_64, glibc 2.39 |
| Audit interpreter | `.venv/bin/python`, Python3.12.3, GCC13.3 |
| Numerical/data | NumPy1.26.4, pandas2.2.2, pyarrow15.0.2, scikit-learn1.4.2, SciPy1.17.1 |
| ML | Torch2.3.0+cpu, Transformers4.40.0, Captum0.7.0, SHAP0.45.1, LIME0.2.0.1 |
| API | FastAPI0.111.0, Uvicorn0.52.4 |
| Frontend | Nodev24.20.0, npm11.19.0, Vite5.4.21; existing node_modules used |
| Missing optional audit dependencies | pytest, BERTopic and sentence-transformers not installed in this venv |
| Version caveat | Root requirement pins include SciPy1.13 and Uvicorn0.29, differing from current installed versions; BERTopic saved run config records Torch2.7.1+cu128 |

These are current verification versions, not the complete historical training environment. No installation was attempted. Saved training hardware is recorded separately in the dossier.

## Checks actually performed

| Check / command or method | Outcome | Limitation |
| --- | --- | --- |
| `pwd`, `git status --short`, `git log` and `rg --files`/targeted `rg -n` | Baseline clean; file/component inventory recovered | Git does not prove contribution ownership |
| Search `AGENTS.md` and `BSC_THESIS_PROJECT_GUIDE.md` in repository/inspected parents | Neither located | Missing, not inferred instruction |
| Read relevant READMEs/config/train/evaluation/runtime/frontend/test/deployment code; notebook JSON source inspection | Active/legacy/proposed paths and run differences traced | Notebooks were not executed |
| `pdftotext -layout` on formatting PDF and proposal into `/tmp`; DOCX zip `word/document.xml` paragraph extraction | All formatting/proposal text and template contents read | Template layout not rendered as a new thesis; source documents not modified |
| `pdftoppm` selected formatting pages to scratch; view pages1 and4 | Visual format/structure inspection | Not a submission-layout validation |
| Primary-paper/publisher/official-model web searches/opens | Bibliography corrections and MentalBERT NC tag verified | Targeted lookup, not systematic review; publisher [12] direct403 and [2] proceedings metadata unresolved |
| `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m unittest discover -s test/c3_text_stressor_distortion -p test_runtime_contract.py -v` | Five tests pass | Preprocessing/config/calibration/mock aggregation; no real model load/inference |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=agent/backend .venv/bin/python -B -m unittest discover -s agent/backend/tests -p test_fusion.py -v` | Ten tests pass | Pure shared fusion, not full portal/agent/browser validation |
| `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B /tmp/c3_audit.py` | Counts/fields/distributions,36 overlap comparisons,18 normalization checks,101 AST parses,273 SHA-256 hashes | Temporary read-only audit; no values of sensitive text/keys recorded |
| `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B /tmp/c3_metrics.py` | Sixteen saved model-metric comparisons agree at1e-9; calibration metrics/mapping and ensemble arithmetic checked | Recomputed saved predictions, not regenerated inference |
| Lightweight S-BEST label/prediction/probability array arithmetic using sklearn confusion_matrix/accuracy_score/f1_score/roc_auc_score | Three saved diagnostic reports match their arrays at1e-9; CMs added separately | No checkpoint load; diagnostic/refit/later identities remain distinct |
| Additional in-memory pandas checks shown below | Seed42 calibration partition matches; post overlap/counts; fusion samples counted | Exact grouping recovery only; no near-duplicate search |
| Frontend cwd: `test -d node_modules && npm_config_cache=/tmp/c3-thesis-npm-cache npm run build -- --outDir /tmp/c3-thesis-frontend-build` | Pass:1594 modules, approximately1.57s; HTML.56KB, CSS25.63KB, JS176.46KB | Scratch outDir warning informational; build is not browser acceptance |
| Mock API script with scratch DIARY_DB_PATH, offline HF flags and patched classifiers/topic | TestClient startup blocked; interrupted; 25s bounded diagnostic exited124 with AnyIO/Starlette startup wait | No pass claimed; no actual diary/model touched |
| `timeout 25s env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B /tmp/c3_db_smoke.py` | Thirteen direct SQLite assertions pass with synthetic input/scratch DB, module reload, CRUD/pagination/summary/fusion reconstruction | No HTTP/browser/model inference; module reload is not a full process restart |
| Converted async httpx ASGI mock smoke | Initial scratch-script indentation error corrected; bounded attempt timed out | Transport/persistence remains unresolved; no assertion completion claimed |
| PNG/table inventory initially tried system Pillow | Pillow unavailable; `ModuleNotFoundError` | Corrected using PNG signature/header dimensions, no installation |
| Inventory via pathlib/PNG struct, followed by three aggregate plot views |185 existing image/HTML/table assets inventoried; selected CBT CM/calibration/latency visuals inspected | Text-bearing XAI/diary content not embedded or reproduced |

Exact overlap method: raw string equality and separately `' '.join(s.split()).casefold()`. Count distinct intersection values, participating rows on each side, identical full-row tuple intersections and differing label sets. The thirty-six comparisons cover two tasks × three text variants × two normalizations × three partition pairs. Runtime preprocessing compares saved strings for all three active families in all six partitions (18,204 comparisons); no mismatches. Parse check uses `ast.parse`, not imports/compilation.

Post recovery maps raw Dreaddit training text to sets of `post_id`, joins by raw text without exposing identifiers, and uses single-post mappings for confirmed train/validation overlap (143); ambiguous mappings yield153 possible. Raw train/test posts have no intersection. Calibration partition reconstruction calls `train_test_split` on positional indices with test_size0.20, random_state42 and binary stratification, and matches stored calibration indices for both tasks. Metric code checks original_test_index uniqueness/label order, maps calibrated probabilities, recomputes all CMs and appropriate positive/macro metrics, and compares matched JSON keys at absolute tolerance1e-9. No raw sensitive inputs are needed in the dossier.

## Reproducible audit code

The following scratch scripts are included as text so the dossier does not depend on surviving `/tmp` files. To rerun, copy them into the named scratch paths and run the commands above from repository root, adjusting ROOT for another checkout. They overwrite only `/tmp/c3_audit.json`. The second script depends on the first. The initial metric discovery used an incorrect suffix and found no metrics; the second corrected the actual saved column schemas before any results were reported. The hash manifest below is the present artifact baseline. Avoid importing the full application without first setting a scratch DIARY_DB_PATH.

### Scratch data/hash audit (`/tmp/c3_audit.py`)

```python
import csv, json, pathlib, itertools, collections, importlib.util, sys, math, hashlib, ast
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score, average_precision_score, confusion_matrix, balanced_accuracy_score
ROOT=pathlib.Path('/mnt/storage/SLIIT/research/R26-IT-036')
out={'datasets':{},'overlap':[],'runtime_preprocessing':[], 'metrics':[], 'calibration':[], 'syntax':{},'hashes':{}}
sys.path.insert(0,str(ROOT))
from backend.c3_text_stressor_distortion.app.preprocessing import preprocess_stress_text, preprocess_cbt_text
from backend.c3_text_stressor_distortion.app.calibration import calibration_metrics, apply_temperature
from backend.c3_text_stressor_distortion.app.fusion.decision import decide_fusion
def read(p):
 with (ROOT/p).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
for task,folder,prefix,textcol,fn in [('stress','Stress header','dreaddit','text',preprocess_stress_text),('cbt','CBT header','cbt','Patient Question',preprocess_cbt_text)]:
 frames={}
 for split in ['train','val','test']:
  p=f'data/{folder}/processed/{prefix}_{split}.csv'; rows=read(p);frames[split]=rows
  counts=dict(sorted(collections.Counter(int(r['label']) for r in rows).items()))
  texts=collections.Counter(r[textcol] for r in rows); binary=collections.Counter(int(int(r['label'])!=0) for r in rows)
  out['datasets'][p]={'rows':len(rows),'fields':list(rows[0]),'labels':counts,'binary':dict(sorted(binary.items())),'empty_text':sum(not r[textcol].strip() for r in rows),'distinct_raw_text':len(texts),'extra_raw_text_rows':sum(n-1 for n in texts.values()),'extra_identical_rows':len(rows)-len({tuple(r.items()) for r in rows})}
  if task=='cbt':out['datasets'][p]['secondary_nonempty']=sum(bool(r.get('secondary_distortion','').strip()) for r in rows)
  for model,col in [('BERT','text_for_bert'),('MentalBERT','text_for_mentalbert'),('DeBERTa-v3','text_for_deberta')]:
   mismatches=0;empty=0
   for r in rows:
    try:value=fn(r[textcol],model)
    except ValueError:value='';empty+=1
    mismatches+=value!=r[col]
   out['runtime_preprocessing'].append({'path':p,'model':model,'mismatches':mismatches,'empty':empty,'n':len(rows)})
 for col in [textcol,'text_for_bert','text_for_deberta']:
  for norm in ['exact','strip_collapse_whitespace_casefold']:
   def normal(x):return x if norm=='exact' else ' '.join(x.split()).casefold()
   for a,b in itertools.combinations(frames,2):
    ca=collections.Counter(normal(r[col]) for r in frames[a]); cb=collections.Counter(normal(r[col]) for r in frames[b]); shared=ca.keys()&cb.keys()
    same={tuple(r.items()) for r in frames[a]}&{tuple(r.items()) for r in frames[b]}
    la={x:{r['label'] for r in frames[a] if normal(r[col])==x} for x in shared};lb={x:{r['label'] for r in frames[b] if normal(r[col])==x} for x in shared}
    out['overlap'].append({'task':task,'column':col,'normalization':norm,'splits':[a,b],'shared_values':len(shared),'rows_a':sum(ca[x] for x in shared),'rows_b':sum(cb[x] for x in shared),'equal_full_row_values':len(same),'shared_values_with_different_label_sets':sum(la[x]!=lb[x] for x in shared)})
raw=read('data/CBT header/raw/Annotated_data.csv')
out['raw_cbt']={'rows':len(raw),'fields':list(raw[0]),'dominant_labels':dict(collections.Counter(r.get('Dominant Distortion','') for r in raw))}
for split in ['train','test']:
 p=f'data/Stress header/raw/Dreaddit/{split}-00000-of-00001.parquet';df=pd.read_parquet(ROOT/p)
 out['datasets'][p]={'rows':len(df),'fields':list(df.columns),'labels':{str(k):int(v) for k,v in df.label.value_counts().items()}}
def calc(y,p,q=None):
 d={'n':len(y),'confusion_matrix':confusion_matrix(y,p,labels=[0,1]).tolist(),'accuracy':accuracy_score(y,p),'f1_macro':f1_score(y,p,average='macro'),'f1_positive':f1_score(y,p),'precision_macro':precision_score(y,p,average='macro'),'recall_macro':recall_score(y,p,average='macro'),'precision_positive':precision_score(y,p),'recall_positive':recall_score(y,p),'mcc':matthews_corrcoef(y,p),'balanced_accuracy':balanced_accuracy_score(y,p)}
 if q is not None:d.update(roc_auc=roc_auc_score(y,q),pr_auc=average_precision_score(y,q))
 return d
for p in ['reports/c3_text_stressor_distortion/CBT header/binary_v4/binary_test_predictions.csv','reports/c3_text_stressor_distortion/CBT header/evaluation/error_analysis/combined_predictions.csv','reports/c3_text_stressor_distortion/Stress header/evaluation/error_analysis/combined_predictions.csv']:
 df=pd.read_csv(ROOT/p);out.setdefault('prediction_schemas',{})[p]={'rows':len(df),'fields':list(df.columns)}
 # Column selection completed after inspecting schemas; never emit text.
 if 'binary_v4' in p:
  y=df['true_binary_label'].to_numpy() if 'true_binary_label' in df else df['binary_label'].to_numpy() if 'binary_label' in df else None
  if y is not None:
   for c in df.columns:
    if c.endswith('_pred'):
     name=c[:-5];q=df[name+'_prob'].to_numpy() if name+'_prob' in df else None
     out['metrics'].append({'path':p,'model':name,**calc(y,df[c],q)})
for task in ['stress','cbt']:
 p=f'reports/c3_text_stressor_distortion/Calibration/{task}/calibration_predictions.csv';df=pd.read_csv(ROOT/p);out.setdefault('prediction_schemas',{})[p]={'rows':len(df),'fields':list(df.columns)}
out['fusion_states']=[{'stress':s,'distortion':d,'state':decide_fusion(s,d).state.value} for s,d in itertools.product([False,True],repeat=2)]
out['syntax']={'checked':0,'failed':[]}
for base in ['backend/c3_text_stressor_distortion','ai_components/c3_text_stressor_distortion','test/c3_text_stressor_distortion','agent/backend/c3']:
 for p in (ROOT/base).rglob('*.py'):
  try:ast.parse(p.read_text());out['syntax']['checked']+=1
  except SyntaxError as e:out['syntax']['failed'].append({'path':str(p.relative_to(ROOT)),'line':e.lineno,'message':e.msg})
for p in list((ROOT/'data').rglob('*'))+list((ROOT/'models/c3_text_stressor_distortion').rglob('*'))+list((ROOT/'reports/c3_text_stressor_distortion').rglob('*'))+list((ROOT/'agent/model/c3').rglob('*')):
 if p.is_file() and ('c3' in str(p) or 'header' in str(p)):
  h=hashlib.sha256()
  with p.open('rb') as f:
   for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
  out['hashes'][str(p.relative_to(ROOT))]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
pathlib.Path('/tmp/c3_audit.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k not in ['hashes','overlap','runtime_preprocessing','datasets']},indent=2))
print('Dataset aggregates and overlap saved without raw text; hashed',len(out['hashes']),'files')

```

### Scratch saved-prediction/calibration audit (`/tmp/c3_metrics.py`)

```python
exec(open('/tmp/c3_audit.py').read().split('for task,folder,prefix')[0])
x=json.load(open('/tmp/c3_audit.json'))
x['metrics']=[];x['metric_comparisons']=[];x['calibration']=[]
for task,p,yp,suffix,psuffix,report in [
 ('cbt_v4','reports/c3_text_stressor_distortion/CBT header/binary_v4/binary_test_predictions.csv','true_binary_label','_prediction','_distortion_probability','reports/c3_text_stressor_distortion/CBT header/binary_v4/all_results.json'),
 ('cbt_later','reports/c3_text_stressor_distortion/CBT header/evaluation/error_analysis/combined_predictions.csv','binary_label','_pred','_prob','reports/c3_text_stressor_distortion/CBT header/evaluation/metrics/all_evaluation_results.json'),
 ('stress_later','reports/c3_text_stressor_distortion/Stress header/evaluation/error_analysis/combined_predictions.csv','label','_pred','_prob','reports/c3_text_stressor_distortion/Stress header/evaluation/metrics/all_evaluation_results.json')]:
 df=pd.read_csv(ROOT/p);y=df[yp].to_numpy();saved=json.loads((ROOT/report).read_text())
 def calc(y,p,q):
  return {'n':len(y),'confusion_matrix':confusion_matrix(y,p,labels=[0,1]).tolist(),'accuracy':accuracy_score(y,p),'f1_macro':f1_score(y,p,average='macro'),'f1_positive':f1_score(y,p),'precision_macro':precision_score(y,p,average='macro'),'recall_macro':recall_score(y,p,average='macro'),'precision_positive':precision_score(y,p),'recall_positive':recall_score(y,p),'mcc':matthews_corrcoef(y,p),'balanced_accuracy':balanced_accuracy_score(y,p),'roc_auc':roc_auc_score(y,q),'pr_auc':average_precision_score(y,q)}
 for c in df.columns:
  if not c.endswith(suffix):continue
  name=c[:-len(suffix)];q=df[name+psuffix].to_numpy();m=calc(y,df[c],q)
  x['metrics'].append({'experiment':task,'path':p,'model':name,**m})
  sk='Weighted_Ensemble' if name=='ensemble' else name
  sm=saved[sk]['test_metrics'] if task=='cbt_v4' else saved[sk]['distortion_detection_metrics' if task=='cbt_later' else 'stress_detection_metrics']
  maps={'f1':'f1_positive','precision':'precision_macro' if task=='cbt_v4' else 'precision_positive','recall':'recall_macro' if task=='cbt_v4' else 'recall_positive','distortion_precision':'precision_positive','distortion_recall':'recall_positive','distortion_f1':'f1_positive'}
  diffs={k:float(m[maps.get(k,k)])-v for k,v in sm.items() if maps.get(k,k) in m and abs(float(m[maps.get(k,k)])-v)>1e-9}
  x['metric_comparisons'].append({'experiment':task,'model':name,'saved_report':report,'mismatches':diffs})
for task in ['stress','cbt']:
 p=f'reports/c3_text_stressor_distortion/Calibration/{task}/calibration_predictions.csv';df=pd.read_csv(ROOT/p);artifact=json.loads((ROOT/f'reports/c3_text_stressor_distortion/Calibration/{task}/calibration_report.json').read_text())
 y=df.true_label.to_numpy();raw=df.raw_positive_probability.to_numpy();cal=df.calibrated_positive_probability.to_numpy();rp=df.raw_prediction.to_numpy();cp=df.calibrated_prediction.to_numpy();part=df.partition.to_numpy();mask=part=='evaluation'
 transformed=apply_temperature(raw,artifact['temperature'],artifact['bias'])
 expected=pd.read_csv(ROOT/('data/Stress header/processed/dreaddit_test.csv' if task=='stress' else 'data/CBT header/processed/cbt_test.csv')).label.to_numpy();expected=(expected!=0).astype(int)
 row={'task':task,'path':p,'partition_counts':dict(collections.Counter(part)),'unique_test_indices':int(df.original_test_index.nunique()),'labels_match_test_order':bool(np.array_equal(y,expected)),'decisions_preserved':bool(np.array_equal(rp,cp)),'max_probability_mapping_error':float(np.max(np.abs(cal-transformed))),'full_test':calc(y,rp,raw),'evaluation':calc(y[mask],rp[mask],raw[mask])}
 row['metrics']={}
 for phase,q in [('before',raw),('after',cal)]:
  m=calibration_metrics(y[mask],q[mask]);m={k:v for k,v in m.items() if k!='reliability_bins'};sm=artifact['evaluation'][phase]
  row['metrics'][phase]=m;row.setdefault('metric_mismatches',{})[phase]={k:float(v)-sm[k] for k,v in m.items() if abs(float(v)-sm[k])>1e-9}
 x['calibration'].append(row)
df=pd.read_csv(ROOT/'reports/c3_text_stressor_distortion/CBT header/binary_v4/binary_test_predictions.csv')
weighted=.1*df.BERT_distortion_probability+.4*df.MentalBERT_distortion_probability+.5*df['DeBERTa-v3_distortion_probability']
x['cbt_ensemble_probability_max_error']=float(np.max(np.abs(weighted-df.ensemble_distortion_probability)))
ca=pd.read_csv(ROOT/'reports/c3_text_stressor_distortion/Calibration/cbt/calibration_predictions.csv');x['cbt_calibration_vs_training_probability_max_error']=float(np.max(np.abs(ca.raw_positive_probability-df.ensemble_distortion_probability)))
rawtrain=pd.read_parquet(ROOT/'data/Stress header/raw/Dreaddit/train-00000-of-00001.parquet');rawtest=pd.read_parquet(ROOT/'data/Stress header/raw/Dreaddit/test-00000-of-00001.parquet')
x['raw_stress_post_group_overlap']={'shared_post_values':len(set(rawtrain.post_id)&set(rawtest.post_id)),'train_posts':rawtrain.post_id.nunique(),'test_posts':rawtest.post_id.nunique()}
paths=[ROOT/f'data/Stress header/processed/dreaddit_{s}.csv' for s in ['train','val','test']];frames=[pd.read_csv(p) for p in paths];tr=set(frames[0].text)|set(frames[1].text);ts=set(frames[2].text);x['stress_refit_test_shared_raw_text_values']=len(tr&ts)
lat=pd.read_csv(ROOT/'reports/c3_text_stressor_distortion/Fusion/rule_coverage_samples.csv');ex=pd.read_csv(ROOT/'reports/c3_text_stressor_distortion/Fusion/explain_latency_samples.csv');x['fusion_csv_schemas']={'predict':list(lat.columns),'explain':list(ex.columns)}
json.dump(x,open('/tmp/c3_audit.json','w'),indent=2)
for m in x['metrics']:print(m['experiment'],m['model'],m['n'],m['confusion_matrix'],round(m['accuracy'],6),round(m['f1_macro'],6),round(m['f1_positive'],6),round(m['roc_auc'],6))
print('COMPARISONS',x['metric_comparisons']);print('CALIBRATION',x['calibration']);print('GROUPING',x['raw_stress_post_group_overlap']);print('REFIT OVERLAP',x['stress_refit_test_shared_raw_text_values']);print('CBT PROBABILITY ERRORS',x['cbt_ensemble_probability_max_error'],x['cbt_calibration_vs_training_probability_max_error']);print('LATENCY SCHEMAS',x['fusion_csv_schemas'])

```

### Scratch direct SQLite verification (`/tmp/c3_db_smoke.py`)

```python
import os,sys,tempfile,importlib
sys.path.insert(0,'/mnt/storage/SLIIT/research/R26-IT-036')
with tempfile.TemporaryDirectory(prefix='c3-thesis-db-') as scratch:
 os.environ['DIARY_DB_PATH']=scratch+'/synthetic.sqlite3'
 from backend.c3_text_stressor_distortion.app import db
 checks=0
 def check(v):
  global checks
  assert v
  checks+=1
 db.init_db();check(db.summary()['total_entries']==0)
 entry=db.insert_entry(content='Synthetic persistence fixture.',title='Synthetic',stress_label='Stressed',stress_is_stressed=True,stress_confidence=.8,stress_probabilities={'Stressed':.8,'Not Stressed':.2},distortion_label='No Distortion',distortion_has_distortion=False,distortion_confidence=.9,distortion_probabilities={'Distortion':.1,'No Distortion':.9})
 eid=entry['id'];check(entry['fusion']['state']=='stress_only');check(entry['topic'] is None)
 check(len(db.list_entries())==1);check(db.list_entries(limit=1,offset=1)==[])
 importlib.reload(db);db.init_db();check(db.get_entry(eid)['content']=='Synthetic persistence fixture.')
 summary=db.summary();check(summary['total_entries']==1);check(summary['stressed_pct']==100 and summary['distortion_pct']==0);check(len(summary['trend'])==14)
 check(db.delete_entry(eid));check(db.get_entry(eid) is None);check(not db.delete_entry(eid));check(db.summary()['total_entries']==0)
 print(f'{checks} direct SQLite assertions passed; scratch only, synthetic input, module reload, no API transport/model inference')
```

### Additional grouping/partition arithmetic (equivalent recorded procedure)

```python
from collections import defaultdict
from sklearn.model_selection import train_test_split
posts = defaultdict(set)
for text, post in rawtrain[['text', 'post_id']].itertuples(index=False, name=None):
    posts[text].add(post)
group_sets = []
possible_sets = []
for frame in frames[:2]:
    group_sets.append({next(iter(posts[t])) for t in frame.text if len(posts[t]) == 1})
    possible_sets.append(set().union(*(posts[t] for t in frame.text)))
confirmed = len(group_sets[0] & group_sets[1])
possible = len(possible_sets[0] & possible_sets[1])
for task in ['stress', 'cbt']:
    df = pd.read_csv(ROOT / f'reports/c3_text_stressor_distortion/Calibration/{task}/calibration_predictions.csv')
    evaluation, calibration = train_test_split(np.arange(len(df)), test_size=.20,
        random_state=42, stratify=df.true_label.to_numpy())
    assert set(calibration) == set(df.loc[df.partition == 'calibration', 'original_test_index'])
# Count numeric/state columns only; do not emit sample text or identifiers.
```

## Checks intentionally not run and commands for a future authorised task

Commands below are documentation, **not executed** here. They can train, invoke substantial inference, download assets, create a database or overwrite outputs. Review CLI/config and use a new versioned output destination before execution; do not paste them over existing result directories.

| Activity | Entry point / command to inspect and run only after authorisation | Why not run |
| --- | --- | --- |
| Stress training/tuning/CV/refit | `python 'ai_components/c3_text_stressor_distortion/Stress header/train/train.py'` | Training; current default output may overwrite saved artifacts; checkpoint/scheduler defects first |
| CBT binary training | `python 'ai_components/c3_text_stressor_distortion/CBT header/train/binary_classification/train.py'` or its `train_modal.py` entry | Training/paid-service possibility; recover invocation and overrides first |
| BERTopic fitting | Inspect `ai_components/c3_text_stressor_distortion/Stress_header_BERTopic/` and `CBT_header_BERTopic/` run entry/config | GPU/embedding fitting and artifact writes |
| Calibration regeneration | `python ai_components/c3_text_stressor_distortion/calibration/evaluate_and_fit.py` (inspect CLI first) | Can infer models/write current artifacts; historical test reuse and cached provenance first |
| Real handcrafted tests/reports | `python test/c3_text_stressor_distortion/generate_report.py`; `python test/c3_text_stressor_distortion/generate_ensemble_charts.py` | Loads current models and overwrites demonstration reports |
| Current fusion/XAI timing | `python ai_components/c3_text_stressor_distortion/Fusion/evaluation/rule_coverage_and_latency.py` (inspect CLI/output first) | Expensive XAI and report writes; not needed to verify saved numbers |
| Real-model XAI examples | S/C `xai/run_xai.py` entry points | Potential expensive inference and text-bearing artifacts |
| Full notebook execution | S/C EDA/preprocessing/evaluation notebooks | Preprocessing/evaluation cells can overwrite datasets/results or load models |
| Public/local server | `uvicorn app.main:app --app-dir backend/c3_text_stressor_distortion --port 8005` | Would initialize default diary DB unless overridden; browser/real inference not authorised for this audit |
| Full shared agent suite/deployment | `agent/backend/tests/`, `agent/backend/README.md`, `agent/DEPLOYMENT.md` | Dependencies/models/services/security context exceed lightweight C3 check |
| New group-aware, external, human or clinical evaluation | No single validated script/protocol yet established | Requires design/permissions; not a retrospective invented result |

## Present artifact manifest

SHA-256 was streamed without unpickling/loading models.273 files total9,371,743,932 bytes across relevant local data, C3 research models/reports and integrated model copies. This identifies the files inspected now; it is not a generation-time manifest. Five deployed research/integrated checkpoint pairs match by full hash. No credentials/databases or raw text values appear below.

| Repository-relative artifact | Bytes | SHA-256 |
| --- | --- | --- |
| `agent/model/c3/CBT/BERT_binary_final.pt` | 438811074 | 91ba8796aa14aae6da377029f1c01eefacee82eea3d9fab6bfc42e3bde2e1b84 |
| `agent/model/c3/CBT/DeBERTa-v3_binary_final.pt` | 736210522 | 70ab98e2bc4a22dfeff207aabfc097bae331683ce0ea773b540217d6fd799a1c |
| `agent/model/c3/CBT/MentalBERT_binary_final.pt` | 438812392 | 689d6417e5d55d1e9e08c615e36624c0bba914b99ee896e1ad03c95c6a6903b4 |
| `agent/model/c3/CBT/binary_config.json` | 2532 | 4eeb6662167070a0c55df7b4d2ecf0c090799e45ff169e1314a3eb08e9345257 |
| `agent/model/c3/CBT/binary_ensemble_calibration.json` | 8441 | 81cb0707abddc8722dae5fc009103cac6b740d32a3f40c9853d6c0cd949112f8 |
| `agent/model/c3/CBT/cbt_preprocessing.ipynb` | 183646 | e638db4acbc663a5a9dc24444d49ab5ad88f4b604bbfa92bfdcd8e0bc37f2111 |
| `agent/model/c3/CBT/metadata.json` | 2951 | cc98f9ab715beb9250dfd54674329a3ac0facdf60de7244dce3619a70c69a6f9 |
| `agent/model/c3/base_models/bert-base-uncased/config.json` | 570 | 7160e1553ad2ca51d8c1cb066be533db31826e12d173824c1bb0cb1a4f187d20 |
| `agent/model/c3/base_models/bert-base-uncased/tokenizer.json` | 466062 | ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98 |
| `agent/model/c3/base_models/bert-base-uncased/tokenizer_config.json` | 48 | a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10 |
| `agent/model/c3/base_models/bert-base-uncased/vocab.txt` | 231508 | 07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3 |
| `agent/model/c3/base_models/deberta-v3-base/config.json` | 579 | 649f6a1ec33c6bdd9a6486d5c66019d461139e54957073eafe9bbc2d34c75b0b |
| `agent/model/c3/base_models/deberta-v3-base/spm.model` | 2464616 | c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd |
| `agent/model/c3/base_models/deberta-v3-base/tokenizer_config.json` | 52 | 3f3978e0c036f2c2588cac34a6047cbb0af0b0dc1814254e291028529805496d |
| `agent/model/c3/stress/BERT_best.pt` | 439608442 | 241c588044d2a1a229d52a4cd3b175825fb94105cf3b7be167a1245dbde6fe95 |
| `agent/model/c3/stress/BERT_final.pt` | 439608655 | 1233f8fb943d56e578965fc10f8e29103714f61e74577b01ef709d49a5a07ded |
| `agent/model/c3/stress/DeBERTa-v3_best.pt` | 737007922 | 061d6ca5f1f8292d42155ecd68d65b3debe51449727bc85d17a68a3f1aebc11b |
| `agent/model/c3/stress/DeBERTa-v3_final.pt` | 737008134 | 6f79958c980a758c3c5ee3d4961f61b0b617a5d421dc258e188fa6e8884660d6 |
| `agent/model/c3/stress/dreaddit_preprocessing.ipynb` | 100335 | c1d832eb35aa0a870e113f2d43765fb2011bbb44e65a2a0e351a1bddf570310a |
| `agent/model/c3/stress/metadata.json` | 2085 | f948ee27c02f4a4c7adc26534dce84f176fcba798c6f1afe148a6882d32de387 |
| `agent/model/c3/stress/stress_ensemble_calibration.json` | 8455 | bfe1826a0def1f2d0437e1b9336811e68bcbd56eb7ea38616ad698fc0c1470f6 |
| `data/CBT header/processed/cbt_test.csv` | 2233351 | 51519437e4884bfdc4a0f70463d3ad103b654933bbc341315e09d6448190915a |
| `data/CBT header/processed/cbt_train.csv` | 7291960 | 452caf8ea1259f3ad5c1a43ae2eb85d9b36c030b3bc83f3d1722a8fdeb6c3ce9 |
| `data/CBT header/processed/cbt_val.csv` | 1787653 | db0733149dafbdeaef81e06c4e986859dba3f8256c4dcd59c9f7aaaf4312fa43 |
| `data/CBT header/processed/metadata.json` | 2951 | cc98f9ab715beb9250dfd54674329a3ac0facdf60de7244dce3619a70c69a6f9 |
| `data/CBT header/raw/Annotated_data.csv` | 2604907 | b3a4ca0ee37c9a9ccb5a640d134b59b29f0d5b7afe488a5741bd32b58ea62e90 |
| `data/CBT header/raw/README.md` | 5942 | 485c07d3bdb805b47ba2510f4f827c4fe9dbf5da31832a61fe606f8b0b2f69ec |
| `data/CBT header/raw/Therapist_responses.csv` | 11538743 | 7ed4b5841a15a8d2aff3111ed6184f46ebda88e645a868927c8c725b28417331 |
| `data/Stress header/README.md` | 4895 | a134639f5563a6ad94919ba63a3111adf50490113fc61f81ef62913f61af7155 |
| `data/Stress header/processed/dreaddit_test.csv` | 1301219 | 261f4000cacdf55af04fb756cfa6f851cbfcf14597bf90ef8579f5b3fab9b3c2 |
| `data/Stress header/processed/dreaddit_train.csv` | 4121723 | 1c63d1a7b1d1cf135cb104c687b6c952dbac527681dda5f38b855fcfbfbe529d |
| `data/Stress header/processed/dreaddit_val.csv` | 1020551 | 7108195822812b54617e5c823beed5cf4bf452900b673527cf146867df9e5831 |
| `data/Stress header/processed/metadata.json` | 2085 | f948ee27c02f4a4c7adc26534dce84f176fcba798c6f1afe148a6882d32de387 |
| `data/Stress header/raw/Dreaddit/test-00000-of-00001.parquet` | 492945 | e190c6be8ad9dee717fee11c4756b935b2996566cc5acbfd86252caa2477bf4b |
| `data/Stress header/raw/Dreaddit/train-00000-of-00001.parquet` | 1804928 | 8f855d163755c21d1fcc955fd46e45f730b97ecce4d3380f84da922cf65369a7 |
| `models/c3_text_stressor_distortion/CBT header/BERT_binary_final.pt` | 438811074 | 91ba8796aa14aae6da377029f1c01eefacee82eea3d9fab6bfc42e3bde2e1b84 |
| `models/c3_text_stressor_distortion/CBT header/BERTopic/config.json` | 358 | c04ffd0ea7e434c936cf88b4112ad51c35c42394fb5ea254f482f3fdb70e4697 |
| `models/c3_text_stressor_distortion/CBT header/BERTopic/ctfidf.safetensors` | 70180 | 8e55184b8d6052a96cceb92aaea4847875bb2265dd406b128d843a33ee0ab262 |
| `models/c3_text_stressor_distortion/CBT header/BERTopic/ctfidf_config.json` | 22500 | ea2d491e78582f5155b5992f15ee59f23f64ec8b73fd85b433d43260b3a36730 |
| `models/c3_text_stressor_distortion/CBT header/BERTopic/topic_embeddings.safetensors` | 55384 | 40462f18e3a4c82ab1c69cfddf996421ead5b2ee9bc640f80e783a9cc93f54f0 |
| `models/c3_text_stressor_distortion/CBT header/BERTopic/topics.json` | 21696 | 9bc62314365f55a4871be6f1b2c5cfd6fa27ff9c3126fd498f85f141782ffaa6 |
| `models/c3_text_stressor_distortion/CBT header/DeBERTa-v3_binary_final.pt` | 736210522 | 70ab98e2bc4a22dfeff207aabfc097bae331683ce0ea773b540217d6fd799a1c |
| `models/c3_text_stressor_distortion/CBT header/MentalBERT_binary_final.pt` | 438812392 | 689d6417e5d55d1e9e08c615e36624c0bba914b99ee896e1ad03c95c6a6903b4 |
| `models/c3_text_stressor_distortion/CBT header/binary_baseline_LR.pkl` | 3575649 | 19a7f5f76e42787e8161b7ce8f5a86cf0c01dc91c7119b315081dc5fea3d025e |
| `models/c3_text_stressor_distortion/CBT header/binary_baseline_SVM.pkl` | 4174177 | 40070ae19d483789ce8cfe97d03085d336768ffbdcdf19075f763cf157e3e8de |
| `models/c3_text_stressor_distortion/CBT header/binary_config.json` | 2532 | 4eeb6662167070a0c55df7b4d2ecf0c090799e45ff169e1314a3eb08e9345257 |
| `models/c3_text_stressor_distortion/CBT header/binary_ensemble_calibration.json` | 8441 | 81cb0707abddc8722dae5fc009103cac6b740d32a3f40c9853d6c0cd949112f8 |
| `models/c3_text_stressor_distortion/CBT header/ensemble_selection.json` | 9186 | 5093ecc43959a8c90ec113336bd2c01e8437393c99bd8046b8c747904370c50c |
| `models/c3_text_stressor_distortion/Stress header/BERT_best.pt` | 439608442 | 241c588044d2a1a229d52a4cd3b175825fb94105cf3b7be167a1245dbde6fe95 |
| `models/c3_text_stressor_distortion/Stress header/BERT_final.pt` | 439608655 | 1233f8fb943d56e578965fc10f8e29103714f61e74577b01ef709d49a5a07ded |
| `models/c3_text_stressor_distortion/Stress header/BERTopic/config.json` | 358 | c04ffd0ea7e434c936cf88b4112ad51c35c42394fb5ea254f482f3fdb70e4697 |
| `models/c3_text_stressor_distortion/Stress header/BERTopic/ctfidf.safetensors` | 402676 | b77f81b430b2947750ee94b97dd103acc5fdae20e81e54fc291390e638edd31b |
| `models/c3_text_stressor_distortion/Stress header/BERTopic/ctfidf_config.json` | 103879 | cc086cfac2fdf0982481c2115204c62cb3b43040d47993bfe712bb74db77cd7e |
| `models/c3_text_stressor_distortion/Stress header/BERTopic/topic_embeddings.safetensors` | 162904 | 241c1a5e3518cc16b5bb7f516186fbc2daeb91ad27dd51a8397ad77260b220ab |
| `models/c3_text_stressor_distortion/Stress header/BERTopic/topics.json` | 59772 | 192359810ef5a5af9adfe7235d1de098e1b0ea463a28f5223a12901f6da8363d |
| `models/c3_text_stressor_distortion/Stress header/DeBERTa-v3_best.pt` | 737007922 | 061d6ca5f1f8292d42155ecd68d65b3debe51449727bc85d17a68a3f1aebc11b |
| `models/c3_text_stressor_distortion/Stress header/DeBERTa-v3_final.pt` | 737008134 | 6f79958c980a758c3c5ee3d4961f61b0b617a5d421dc258e188fa6e8884660d6 |
| `models/c3_text_stressor_distortion/Stress header/MentalBERT_best.pt` | 439609720 | 520e3e663fa2385391a4704ec05afa15ce4fb9eb00d492c1473e95fe87187231 |
| `models/c3_text_stressor_distortion/Stress header/MentalBERT_final.pt` | 439609933 | e94ed5126c217f648891972288e3acfd3966270b46b21eca905c9301b124c2f9 |
| `models/c3_text_stressor_distortion/Stress header/all_results.json` | 114472 | a57b2a6ed9e790c9f4c262a42e43ef030b9abd5a38a984988838cffcc469f40d |
| `models/c3_text_stressor_distortion/Stress header/baseline_LR.pkl` | 2973137 | 1e678f3821f3e979265b31536ed404f794de980e82304e43820b6cfbc69ee198 |
| `models/c3_text_stressor_distortion/Stress header/baseline_SVM.pkl` | 3456142 | b676b778c4f9d29ab952afc5bc2065d4c7187366e0c5294129330255fefb30b3 |
| `models/c3_text_stressor_distortion/Stress header/stress_ensemble_calibration.json` | 8455 | bfe1826a0def1f2d0437e1b9336811e68bcbd56eb7ea38616ad698fc0c1470f6 |
| `models/c3_text_stressor_distortion/Stress header/training.log` | 2624 | b12268becfa727e615395925bba307478a65fabdc50560b4efebaa209ef8bf8e |
| `reports/c3_text_stressor_distortion/CBT header/BERTopic/dataset_summary.json` | 2104 | 5417d6c49af7bf3d8ae35c1d5b8bc5975c4ed8a475a11026e81ae9e6d1879e93 |
| `reports/c3_text_stressor_distortion/CBT header/BERTopic/document_topic_probabilities.npy` | 344072 | 52260c4476977cf6feb199325cb77d076ece214212bd3ad24f186cef211b41e8 |
| `reports/c3_text_stressor_distortion/CBT header/BERTopic/document_topics.csv` | 3960297 | 41f9e84a150ff37bf15f7fb5c307ec787f902cf6f3784e309e80ae90a95a8051 |
| `reports/c3_text_stressor_distortion/CBT header/BERTopic/fit_filter_summary.json` | 309 | bd6319f5f7cdd2687706d0a1bc98c6b83fea471b085758aaeeb8cbe85c750499 |
| `reports/c3_text_stressor_distortion/CBT header/BERTopic/plots/topic_barchart.html` | 3650594 | 610a7edee3b8a1c009d99cddade0919e780e208e69249e62a2a60fa077d6945f |
| `reports/c3_text_stressor_distortion/CBT header/BERTopic/plots/topic_distortion_status_distribution.png` | 116468 | cbdfc8b512abe4a2afaed0aedf917c416a16ac6f338ed021fd7aca5889deb611 |
| `reports/c3_text_stressor_distortion/CBT header/BERTopic/plots/topic_distortion_type_distribution.png` | 317874 | 6c0dda9f9105b79153c2d8a9c472626d18e0c405286115df66b6241c8586584a |
| `reports/c3_text_stressor_distortion/CBT header/BERTopic/plots/topic_heatmap.html` | 3647639 | d1304b4952ab71292ec257dc9a5864641b2572c9e852d55b9118ddf3e202d4ce |
| `reports/c3_text_stressor_distortion/CBT header/BERTopic/plots/topic_hierarchy.html` | 3644899 | 1a4745b0b52354b3868f7c19a7f1270c9d1bd63742b6744f5fa466f0227e57a7 |
| `reports/c3_text_stressor_distortion/CBT header/BERTopic/plots/topic_sizes.png` | 213481 | bf84f83d6b7571d41a159dae0f19c84be0943ceb35091965558ea4c68f85f0c8 |
| `reports/c3_text_stressor_distortion/CBT header/BERTopic/plots/topics.html` | 3648941 | bc5694d9b1cf8c579319c2d30a63e1f9b4d8089810eec7ed8cf17809de6df05c |
| `reports/c3_text_stressor_distortion/CBT header/BERTopic/probability_matrix_metadata.json` | 285 | c96bbf36fb21a64b02dcb7cb12d28198fa4a5e7146923b8a153a60e949b132e0 |
| `reports/c3_text_stressor_distortion/CBT header/BERTopic/run_config.json` | 2171 | 18ce0980777500f1f220c95e83bd8c14969116e4b978b17e0c300b5f6a4b8e09 |
| `reports/c3_text_stressor_distortion/CBT header/BERTopic/topic_by_distortion_status.csv` | 2667 | 6a0d3e98c665fbbbacaad65c943772517e3558b4c1e013808930a526ed50025d |
| `reports/c3_text_stressor_distortion/CBT header/BERTopic/topic_by_distortion_type.csv` | 14298 | 1bce13dd70b2d95498361b95189c1350f6b48270fcd693b07cfda83e38ce8b39 |
| `reports/c3_text_stressor_distortion/CBT header/BERTopic/topic_info.csv` | 33196 | c60177a6fcf0d8b639cf84af127ad8c60fb810fd96e0e9ed096a99e36f18a99b |
| `reports/c3_text_stressor_distortion/CBT header/BERTopic/topic_metrics.json` | 309 | dbfe259f61fa421547022d17ab8d32ae663ab59b28adf0450d3d8ccfdbdd6161 |
| `reports/c3_text_stressor_distortion/CBT header/BERTopic/topic_terms.csv` | 6002 | b4f3dc19dbc8337a9e0120f4ed2a4c779c15a9a2331de927e2564bf6960089fd |
| `reports/c3_text_stressor_distortion/CBT header/BERTopic/training.log` | 1374 | 8dacdffa53680135dd612aff08a80f472929d9962363228bb2e0696faf430763 |
| `reports/c3_text_stressor_distortion/CBT header/binary_v4/all_results.json` | 31493 | eb04c60124c28879ee7cbffc8bd82599aa21c51fdf54b91faa12bf19866b2c3b |
| `reports/c3_text_stressor_distortion/CBT header/binary_v4/binary_test_predictions.csv` | 65437 | 35ce8722da1a038669328a3d3e4d5007f69dc0ec1656d4f1ed26c76ff82c4f62 |
| `reports/c3_text_stressor_distortion/CBT header/binary_v4/comparison_dashboard.png` | 90355 | 028589b5e89262e110b55b2095fe89ead14eb4bc529c62952e52ddf1752482c1 |
| `reports/c3_text_stressor_distortion/CBT header/binary_v4/confusion_matrices.png` | 104388 | 303f5a0d55dd2bf2cd4c50e4a8e65079f8ef65c8933bf22942a96d472695c8a9 |
| `reports/c3_text_stressor_distortion/CBT header/binary_v4/training.log` | 25889 | 35fa32609f9be80669f77b92e32801e4445c94998d58bbd6649f6c99aa88d37d |
| `reports/c3_text_stressor_distortion/CBT header/binary_v4/training_curves.png` | 203199 | 6d6760d04e5ed859c9cc310b8389d923bc9a6bdd00cd92c62500c7298a029d31 |
| `reports/c3_text_stressor_distortion/CBT header/eda_outputs/cbt_linguistic_features.png` | 250039 | 0662fc44d625b57d643d827fb4bd1469c85c5e09fe6201ff2888e426c16c1a59 |
| `reports/c3_text_stressor_distortion/CBT header/eda_outputs/class_balance.png` | 162010 | f05753b9d3afdb0f74c88961902ab953e104f98e3a9f623fa6ea179421a4910e |
| `reports/c3_text_stressor_distortion/CBT header/eda_outputs/correlation_heatmap.png` | 424134 | 15a948c94c7f7724e733af1fc0b1e0c5c19276d644428fabb5cfa5954b2f2fe2 |
| `reports/c3_text_stressor_distortion/CBT header/eda_outputs/distorted_span_analysis.png` | 107188 | 9c3d0fe44c0542563780a21ec228580717845f8af9201c94be7fafe2c93c0065 |
| `reports/c3_text_stressor_distortion/CBT header/eda_outputs/distortion_cooccurrence.png` | 154167 | fd96e7ffc4bbbab568991007541e898b474f35df351ed9d63d6c4bad250c9c93 |
| `reports/c3_text_stressor_distortion/CBT header/eda_outputs/text_length_features.png` | 262610 | ad3d5cc817a862d457dc4610716dde5041a0d277bea3e13fea8a6c27a27cc08d |
| `reports/c3_text_stressor_distortion/CBT header/eda_outputs/top_bigrams_by_distortion.png` | 109175 | 8f8279c03edb337814563e7c66d08751b3fd6dbfa9bf95738d970688a9536ab3 |
| `reports/c3_text_stressor_distortion/CBT header/eda_outputs/top_words_by_distortion.png` | 100785 | 57c9c77eaaeefdf8a5d71a80062e4a160f0bdd33d6a4a0a5cc5bb7fac9a417a2 |
| `reports/c3_text_stressor_distortion/CBT header/eda_outputs/writing_style_features.png` | 120079 | ee70e960b6f337142c039dfb2d17582b2dfdab4d79b3a656c50ce63c22f350be |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/error_analysis/agreement_analysis.csv` | 2381991 | 913339811aee691e0c7edcd7e6a31d4ba53aa83eddedab6bf19cb34f488b9a75 |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/error_analysis/combined_predictions.csv` | 2375165 | e7cead398729d3586bc1092ff2b257782e2445fff864b443f3ac69f187dcefb6 |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/error_analysis/distortion_type_error_summary.csv` | 2484 | 32e6133ebb3165d59298ffffcf3020cc7a1182c439fef087e7fee16763a1337f |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/error_analysis/error_analysis_report.txt` | 1620 | 41b08eb5e7b47a210f91f685a4a1af3cd16404d60cfec8da95b179df08ec2ae8 |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/error_analysis/error_analysis_results.json` | 3479 | 2eb4e28a933461414abe2e3bd5a9edbf229461a8b996c8df6531d7d420a8347a |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/error_analysis/error_summary.csv` | 991 | 7655e1c15b0839fc21d6505306a250787382360469af372950f3a003c4375ff0 |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/error_analysis/plots/1_error_rate_comparison.png` | 146246 | 0b6e4dd18d141a34ce3f3c3ec46631c2fb2a5ceba388ba4616f46b3c24f04f13 |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/error_analysis/plots/2_fp_fn_comparison.png` | 132441 | d799e4d6d8b594a764d0da99a1a724ea560a7f0fe65fa0d187000a0549325791 |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/error_analysis/plots/3_confusion_matrices.png` | 232194 | 066e60be2012eb7d693f7cfed8317c424047f2ea70c4d41657dc877838150c97 |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/error_analysis/plots/4_confidence_distributions.png` | 849790 | 71a4bdfb8d735dad9c3c75ba3b2295478ca33b322612a3ff9ddbcd30fa1e6dcb |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/error_analysis/plots/5_error_agreement_matrix.png` | 148779 | fbfe4605bd0d7a871c1cdd343995b75f426c5fdc3f2aed99f027e04df77e6f67 |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/error_analysis/plots/6_error_rate_by_distortion_type.png` | 281260 | f7627bf92e278662fc6fe811a17c481192b86950824bd8d015007f4bb92eb27e |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/metrics/all_evaluation_results.json` | 1626 | 87266330feb71d706c44fc7616e1615f04d81320a4b250f89f419380c9aa283d |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/metrics/detailed_classification_reports.txt` | 2449 | 28b85e407d17f768323f6ea9a750e94052b47b730e6a0c9df4ccd3a3f9330371 |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/metrics/model_comparison.csv` | 713 | 2eb8d124af0cda75527a139e6b53fd258fe42b36b404b47fca9cca6f8856016d |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/metrics/plots/1_metrics_comparison.png` | 206060 | 10bd2bdfb1ccd0042cd00040b9a20318307013e9c96c1cf10302bbecd43b82c9 |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/metrics/plots/2_confusion_matrices.png` | 326119 | e7716a1d1f23a3ccb36b7046e0658df9a0d6be314d806e9ed12a984fad1aa674 |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/metrics/plots/3_roc_curves.png` | 281705 | 472f3cb1194333b294ca1ffc10a8e76c5fe0fcff6c75fc2fc30bf19d3f8a5332 |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/metrics/plots/4_f1_accuracy_comparison.png` | 137805 | be45cabde1424c8d9561744106d251b4eef7d4c1fd30cdc8ab8ab2ff256de8f0 |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/metrics/plots/5_precision_recall.png` | 126860 | 77dcea450f0cdd1143119c96e56d202371255dc9e56abafe6ea030ad18cfb00e |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/metrics/plots/6_metrics_heatmap.png` | 205042 | 805ab08bebf46da40470d5e562cbdbc95e847b52d7880fe5de29662c79e30cd0 |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/xai_outputs/bert/bert_1785677214_ig.png` | 53970 | a0412fdcc99c58a734ff1cb29e64948ea560c8dd3d32f28880b974be00cb9827 |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/xai_outputs/bert/bert_1785677214_report.json` | 839 | 1ab348455f6a1403f2aa6cc25600ab0cf2f61211092e7a135b86e25f446ae83f |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/xai_outputs/deberta_v3/deberta_v3_1785686651_ig.png` | 63761 | 297c0d153095629d28b4e406fe0339be602c6a983c2f1d685cc7e35431d75361 |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/xai_outputs/deberta_v3/deberta_v3_1785686651_report.json` | 1721 | f57be89a712ad60d900c62b377e7884e436ceadcb61e3a72771db5a2c82a2b0f |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/xai_outputs/streamlit/deberta_v3_1785678441_ig.png` | 63739 | f3446403e8f3c13e6897078a2875f6dfad0189bec7149954d0989b26ca3c1551 |
| `reports/c3_text_stressor_distortion/CBT header/evaluation/xai_outputs/streamlit/deberta_v3_1785686799_combined_xai.png` | 65178 | 06857cf65d57ccafad72cc592bfde662f408eab5842d3c4068e087e284b4eead |
| `reports/c3_text_stressor_distortion/CBT header/token_lengths_3models.png` | 103534 | 8138e75a697612f6ad86ec88e20442754c2dd5280f3144b08c996d372c9300cc |
| `reports/c3_text_stressor_distortion/Calibration/cbt/calibration_predictions.csv` | 25818 | e999fda76907e3d6857c5ec5497fbade02f736c27cfe91cfa5d1573ba2e735fa |
| `reports/c3_text_stressor_distortion/Calibration/cbt/calibration_report.json` | 8441 | 81cb0707abddc8722dae5fc009103cac6b740d32a3f40c9853d6c0cd949112f8 |
| `reports/c3_text_stressor_distortion/Calibration/cbt/reliability_diagram.png` | 156922 | a33963d5f43473a546b4fe2454226ccc473d90e4917d65c27a98c5fef86fa676 |
| `reports/c3_text_stressor_distortion/Calibration/stress/calibration_predictions.csv` | 36620 | bba0ba5c8d016114803764de2182c3e265a53eb235f26bbcaa282034706261a5 |
| `reports/c3_text_stressor_distortion/Calibration/stress/calibration_report.json` | 8455 | bfe1826a0def1f2d0437e1b9336811e68bcbd56eb7ea38616ad698fc0c1470f6 |
| `reports/c3_text_stressor_distortion/Calibration/stress/reliability_diagram.png` | 152552 | 473ec2869311d9a6d829e8950bb9933099c3701d92f8158615eb74780a320a2a |
| `reports/c3_text_stressor_distortion/Fusion/README.md` | 722 | 8c310c59d0651f3c6cee6b3ed04ed47d14137a010dab87ebac20128ad999b281 |
| `reports/c3_text_stressor_distortion/Fusion/explain_latency_samples.csv` | 189 | 7fea5dc2085b46f0853a4ae796edd20717455d1969b70e4249cb6d6a80c63322 |
| `reports/c3_text_stressor_distortion/Fusion/latency_report.json` | 637 | 25f354161bf36bb08b96ecf8dfe6c049cf5df59570651c6c1e496f54c73f792b |
| `reports/c3_text_stressor_distortion/Fusion/plots/1_state_distribution.png` | 163040 | 7a79e40d29c438ccca4c62196058bca8f68ffad6724324970b55896b227cfdba |
| `reports/c3_text_stressor_distortion/Fusion/plots/2_predict_latency_distribution.png` | 101411 | 67ede62a601d238f1000c6487d5b1c235ca6f62d25081c59930ddbabfd587812 |
| `reports/c3_text_stressor_distortion/Fusion/rule_coverage_samples.csv` | 49370 | d6789f16ce42e628e0d766e8de96478cdade247e3c62aa67fa3a974d9e65456b |
| `reports/c3_text_stressor_distortion/Fusion/state_distribution.csv` | 118 | 6ed975d4afca7151b52e034b6f9ca8552a5a0a1ea129fe2ed53fe4a4548cc5a6 |
| `reports/c3_text_stressor_distortion/Stress header/BERTopic/document_topic_probabilities.npy` | 1472352 | 617ab8e1fd5e745f58361aa83ac6d5f1946c350192f986bddb3670f9ada6cbc6 |
| `reports/c3_text_stressor_distortion/Stress header/BERTopic/document_topics.csv` | 2117617 | 7072eddb1e024493186b99ef113a2fe8e8987592d99d576d4e3098332dbd2e6a |
| `reports/c3_text_stressor_distortion/Stress header/BERTopic/fit_filter_summary.json` | 234 | 101632f0c09dfdfaf4c10f79c4cf2b022947f8c4238c288a4de84bc572f189c3 |
| `reports/c3_text_stressor_distortion/Stress header/BERTopic/plots/topic_barchart.html` | 3652053 | e57ee2928522223260756a89112edc119e60f79e4d73ae8333a0b4fd46f78b1d |
| `reports/c3_text_stressor_distortion/Stress header/BERTopic/plots/topic_heatmap.html` | 3650160 | 42181fe83d6b8b06c41ff0104b41285cbbc3f484a7a45f44fb1297ab9590c74c |
| `reports/c3_text_stressor_distortion/Stress header/BERTopic/plots/topic_hierarchy.html` | 3645647 | 767a20b5c9fbf9e2c776006c32015944b19f946a07fc0dfb9cdde964c8be8200 |
| `reports/c3_text_stressor_distortion/Stress header/BERTopic/plots/topic_sizes.png` | 268981 | 7bb84082eee84ac0f3c1b99709d6fb67c20607af8c479b9e0dbfbb32d7b159ad |
| `reports/c3_text_stressor_distortion/Stress header/BERTopic/plots/topic_stress_distribution.png` | 308335 | 40fd82ef0024584a81eb1ef66dab0341e837c7fb718a40349041c0280d052584 |
| `reports/c3_text_stressor_distortion/Stress header/BERTopic/plots/topics.html` | 3679562 | bc5528cc723d2dacd2dfb3572697c15ed40e0af7ccb33a47d418ad383fc4c232 |
| `reports/c3_text_stressor_distortion/Stress header/BERTopic/probability_matrix_metadata.json` | 565 | 71e89eb7b7d532ab13586d57872e05c243755ab1e86ef99adbd69e76a9b74b1b |
| `reports/c3_text_stressor_distortion/Stress header/BERTopic/run_config.json` | 1894 | 0ef2090e9003a55d5e1b053a76ae53adb26c2b9514cb8d3481ac01477cf82a38 |
| `reports/c3_text_stressor_distortion/Stress header/BERTopic/topic_by_stress_label.csv` | 8717 | 43f3df1f63502db22c546e2533f89a1e8653ae83e8af7a47eababebebedf611c |
| `reports/c3_text_stressor_distortion/Stress header/BERTopic/topic_by_subreddit.csv` | 17091 | 049a009daabca0af4434d15a2e1d6b0f6c4a6670dd39a143424f3aaef0365baa |
| `reports/c3_text_stressor_distortion/Stress header/BERTopic/topic_info.csv` | 125026 | ef27a9258b6f63fac6cc53fe51d1193d14e968b5666c4ef36ed3f885334bcfb6 |
| `reports/c3_text_stressor_distortion/Stress header/BERTopic/topic_metrics.json` | 309 | 0be4e9e3f9cb66ecc5a658ac5f1d946d19bc6aca5efdb61a85ada00917db09ea |
| `reports/c3_text_stressor_distortion/Stress header/BERTopic/topic_terms.csv` | 18640 | 7b54dc7075c41515cdf76e08520c92d771bb91f92b3db091cbe118d30cd7daa6 |
| `reports/c3_text_stressor_distortion/Stress header/BERTopic/training.log` | 1237 | a9d43e2189d7e5a01bcc9c399d551422c8052bfe69064ccc6fa790f4290bc59a |
| `reports/c3_text_stressor_distortion/Stress header/eda_outputs/class_balance.png` | 38384 | e1bfc792ba1e994f82eeba0bb153ce79b08756d1097e7d27be5ce36660b2d7fd |
| `reports/c3_text_stressor_distortion/Stress header/eda_outputs/sentiment_analysis.png` | 97343 | c3f08aa6154713fdd84359526f519b5f88defcb24807b9ad5a2d6d9b029aefca |
| `reports/c3_text_stressor_distortion/Stress header/eda_outputs/simple_correlation_heatmap.png` | 227212 | c996fb2653d5aaafd5331c8d7c4c30b3ac8ad5ec51f74f212560dda7333dfc1a |
| `reports/c3_text_stressor_distortion/Stress header/eda_outputs/subreddit_counts.png` | 78716 | 518e167e8611125eaf714ac2f391efe794634ee25d0c9035b68e78a37eb153ac |
| `reports/c3_text_stressor_distortion/Stress header/eda_outputs/subreddit_label_distribution.png` | 94329 | 9e42ad1253c3aea4aff02f9946df6ed4797d2026ee9f996ded6ef3f576b7d4f1 |
| `reports/c3_text_stressor_distortion/Stress header/eda_outputs/text_length_features.png` | 119524 | 158a60b06e55754842a1e7754cba53ed3c257f6c15e361579e270cacd616fb5a |
| `reports/c3_text_stressor_distortion/Stress header/eda_outputs/top_bigrams_by_label.png` | 93183 | 7263c2f7b9f5a4a78e4fabc9e7581d273ac7966f69e7321e8f7c73b818dbad5e |
| `reports/c3_text_stressor_distortion/Stress header/eda_outputs/top_words_by_label.png` | 85711 | f956050881bea487784c7fa859281b8b551a70df2f188565546064a53d898baa |
| `reports/c3_text_stressor_distortion/Stress header/eda_outputs/writing_style_features.png` | 48975 | 3b278ec6ab371897a5d4982240e23a9323ebcd33175de900ab0fdd6645ab4a45 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/error_analysis/agreement_analysis.csv` | 1510130 | c45c4a316aa3abb7c2702f6ac0ce151d411bc59cedb753de486e8332aaac1b3b |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/error_analysis/combined_predictions.csv` | 1500491 | e310d25d95af0fcb45c3ef5c86f197c16be6c16f4a17dc4458f58cda5e94f14f |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/error_analysis/error_analysis_report.txt` | 1081 | e1d28f2812afbc35e35105ef8f9a4d1d317158d10b33cf93c12c14fd4d7bcfd0 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/error_analysis/error_analysis_results.json` | 3967 | bfffce303fc4a7579be0fc8ee02ee2a408b57eae9d82aaef805b44fa2f97e32a |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/error_analysis/error_summary.csv` | 993 | fdbe6d217210a1bb11f030c8dcf987d0429490e1a34efcc64bdc94f36a729efa |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/error_analysis/plots/1_error_rate_comparison.png` | 161361 | 7cc17064c296acd19c1275cf0451fb1047516dbacb3de8d45fbc15fc52366e2c |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/error_analysis/plots/2_fp_fn_comparison.png` | 145933 | c00119c9f72795bad565cd4a01f96ec4096211baaa17c21b67b3b99156677dc4 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/error_analysis/plots/3_confusion_matrices.png` | 246069 | 3ec14225eb7dd9f67a7bd15dbd9aac6f2e8c2d8889323305f3f3fd14be716bad |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/error_analysis/plots/4_confidence_distributions.png` | 785552 | f1b40154cfe79cb488400440c2e1a55dd99a4bce3767e374bcd590a85e4ff2f3 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/error_analysis/plots/5_error_agreement_matrix.png` | 162188 | 8cdead99d59989e9e036ed929fe2f0b4a7a8af39c9106fd6c177b9c3c9aafea5 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/error_analysis/subreddit_error_summary.csv` | 214 | 8f077889f732cc55c5354beba69d6ec87669a481b64de92aa8895ebd8fd7bea2 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/metrics/all_evaluation_results.json` | 1864 | 868efb5cfac0a08c00c4e8e8b4456026ef1e15b4e53fb1b1a60a434618146015 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/metrics/detailed_classification_reports.txt` | 2419 | d9536608d4c2d1095d10d312406e0d9e12f7c40329695cad43991bc19701bd18 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/metrics/model_comparison.csv` | 715 | ef63ebfc1bbff2078a4c275541144e6893470c3428d2a59bb3696b94e50e6bb4 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/metrics/plots/1_metrics_comparison.png` | 210530 | 31ebf44778dccad9cf7d7db406b579832ed1def19f7ea4bedca178c3d1c5d94c |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/metrics/plots/2_confusion_matrices.png` | 338333 | 6adaf70889a864f88cbe0416dd140e43fc1d09e3914f24208a807329850e8c5a |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/metrics/plots/3_roc_curves.png` | 291862 | cb97934637c1932e0fbb87715e716e80bada78df22138b7dfe11418e3e484b5d |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/metrics/plots/4_f1_accuracy_comparison.png` | 136881 | 28c59f1490699b3d57a0a43f5780f12653ab731743b2537fedd168ea26d1f70d |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/metrics/plots/5_precision_recall.png` | 123691 | 1c9b012cb701c0ee8e978a6a38468cb6527d1f7fdbe11ec7118042bdbac54f65 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/metrics/plots/6_metrics_heatmap.png` | 206449 | 69d472e998298e4fe453377bdf8f352d54aac7858195494600afdfef60f0046b |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/metrics/plots/BERT_best.pt` | 439608442 | 241c588044d2a1a229d52a4cd3b175825fb94105cf3b7be167a1245dbde6fe95 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/metrics/plots/comparison_dashboard.png` | 89564 | b39c14391f8ff09b691ec0f8f41a619c9bf9e5f275f1f42f79ae8070007c08c4 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/metrics/plots/confusion_matrices.png` | 107825 | 943fd6c950451aa385a9b0ae4b1c151b8d3551cb9e740660ad329bf66dc7a387 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/metrics/plots/training_curves.png` | 213149 | f0074776de1334da73d8af524978e1678dfa3a650dc9e9fc5d6c0ea2f3cd8b1d |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/metrics/subreddit_classification_metrics.csv` | 166 | b0d222793a0e22cbb066099986167d5fc6d65468d19c7a11f9230423fbc0d566 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/ig_plots/ig_sample.png` | 52265 | 03e4780744cf7715f4b9072f9634c05ba5dbff965652717782230c4bb502d1f7 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/shap_plots/shap_sample.html` | 53928 | 6737fd9c424797c819ab1f0a0d4de78d665a344ec88dd1143fed83ef20c88d14 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/shap_plots/shap_sample.png` | 46753 | 844b6d5ececf5474e1e421caad6a4dd033dfca7dedc822fcb1500d467c15c5d9 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/bert_1779863614_ig.png` | 71488 | 1a8623c9c06ab41b097133b0f90b75cf959e42c22ca07180c4c66465331c35c7 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/bert_1779863614_shap.html` | 112311 | b902a34e0669a0c8d5b4846ebf426d9630472940b5c922d34059c032259ca1a9 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/bert_1779863614_shap.png` | 71753 | ebe6baef0d5a884426587b1c75f69859495fe637a7cbc1690057f9ade87e2ab2 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/bert_1779863935_ig.png` | 71488 | 1a8623c9c06ab41b097133b0f90b75cf959e42c22ca07180c4c66465331c35c7 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/bert_1779863935_shap.html` | 112311 | 7e4d2fa6d4ed658e297bd82aed1133c60b2cd8e509abda5bdef0ede2dd2efeb2 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/bert_1779863935_shap.png` | 71753 | ebe6baef0d5a884426587b1c75f69859495fe637a7cbc1690057f9ade87e2ab2 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/bert_1779863993_ig.png` | 71488 | 1a8623c9c06ab41b097133b0f90b75cf959e42c22ca07180c4c66465331c35c7 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/bert_1779863993_shap.html` | 112311 | 8a84891c02690436c789dcbd7c3662c29bb9d1b873f547b74f8fac2537712b80 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/bert_1779863993_shap.png` | 71753 | ebe6baef0d5a884426587b1c75f69859495fe637a7cbc1690057f9ade87e2ab2 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/bert_1779864150_ig.png` | 71488 | 1a8623c9c06ab41b097133b0f90b75cf959e42c22ca07180c4c66465331c35c7 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/bert_1779864150_shap.html` | 112311 | 84d53ad1a18c6c1abc0572d83777526f83c7b98beb078327b614fb9bd9aeba47 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/bert_1779864150_shap.png` | 71753 | ebe6baef0d5a884426587b1c75f69859495fe637a7cbc1690057f9ade87e2ab2 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779807445_ig.png` | 72921 | 75ef1822012b91e1f09b0f00119c187f2eb9e81471aeb201d9d071fdbb331301 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779807479_shap.html` | 112232 | e5a475fa24e8952f837acf4b79721bffd6220c560370e53efb42ca19037be750 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779807479_shap.png` | 78616 | 46c0c5977ecde300351b87d466ff01cf09d6e28d7d965f0c124e2d243594b8c2 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779854112_ig.png` | 72921 | 75ef1822012b91e1f09b0f00119c187f2eb9e81471aeb201d9d071fdbb331301 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779854162_shap.html` | 112232 | 894a154f843f6766db9def37709ac4e01652c2596c9be5f5a9e4dbaee152b800 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779854162_shap.png` | 78616 | 46c0c5977ecde300351b87d466ff01cf09d6e28d7d965f0c124e2d243594b8c2 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779854211_ig.png` | 72921 | 75ef1822012b91e1f09b0f00119c187f2eb9e81471aeb201d9d071fdbb331301 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779854327_ig.png` | 72921 | 75ef1822012b91e1f09b0f00119c187f2eb9e81471aeb201d9d071fdbb331301 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779854346_shap.html` | 112232 | 42e8d9906037e8677ffd44cf56b46e5c6332c243118fdfe7bd81bad6665aadc1 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779854346_shap.png` | 78616 | 46c0c5977ecde300351b87d466ff01cf09d6e28d7d965f0c124e2d243594b8c2 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779854395_ig.png` | 72921 | 75ef1822012b91e1f09b0f00119c187f2eb9e81471aeb201d9d071fdbb331301 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779854452_shap.html` | 112232 | fa4550f4e3fd642847ca1ea7db1ce6ecd27879412712c37c87b3620ba55a08a8 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779854452_shap.png` | 78616 | 46c0c5977ecde300351b87d466ff01cf09d6e28d7d965f0c124e2d243594b8c2 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779854535_ig.png` | 72921 | 75ef1822012b91e1f09b0f00119c187f2eb9e81471aeb201d9d071fdbb331301 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779854568_shap.html` | 112232 | 161d1821a668e4afe7a6dcfa7b8c02eb8579b2bee01f6429a0ddcaa4c45a0fdf |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779854568_shap.png` | 78616 | 46c0c5977ecde300351b87d466ff01cf09d6e28d7d965f0c124e2d243594b8c2 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779854692_shap.html` | 112232 | 37c4d925c5fafc6747169504c2d66901085276f9413ad9c1bf60f3f6964cc4fe |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779854692_shap.png` | 78616 | 46c0c5977ecde300351b87d466ff01cf09d6e28d7d965f0c124e2d243594b8c2 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779855735_ig.png` | 72921 | 75ef1822012b91e1f09b0f00119c187f2eb9e81471aeb201d9d071fdbb331301 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779855749_shap.html` | 112232 | 5387ce35d9d0aca14bd9dfd7128688d531fda777e048787493af5cd4204b0418 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779855749_shap.png` | 78616 | 46c0c5977ecde300351b87d466ff01cf09d6e28d7d965f0c124e2d243594b8c2 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779856673_attention_rollout.png` | 63321 | af9eb8f00c459c497cfab31aa3f173b9ac323a30e1ca6cbe3196eeef2a5e0ca8 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779862660_ig.png` | 72921 | 75ef1822012b91e1f09b0f00119c187f2eb9e81471aeb201d9d071fdbb331301 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779862660_shap.html` | 112232 | c67be42992fba2fbc1e7beee9ad22d49014ca35105c43fdd57a0ad0f7e83b73a |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1779862660_shap.png` | 78616 | 46c0c5977ecde300351b87d466ff01cf09d6e28d7d965f0c124e2d243594b8c2 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1783734346_ig.png` | 72921 | 75ef1822012b91e1f09b0f00119c187f2eb9e81471aeb201d9d071fdbb331301 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1783828215_ig.png` | 73035 | 0a3b00b07d2e4ac90d4f638772d3aa39372278b3d182d3053e849f7f5921a68f |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1783828287_ig.png` | 73035 | 0a3b00b07d2e4ac90d4f638772d3aa39372278b3d182d3053e849f7f5921a68f |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1783828287_shap.html` | 112695 | c327e652f7a8fc5229799238174a87d5247592635cb4af8ea510df817158110e |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1783828287_shap.png` | 77390 | c6e9cfe9bb5a4011298e84162f59aa49f771d07d8a2d12d543384f8644ed00b1 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1784170784_ig.png` | 73035 | 0a3b00b07d2e4ac90d4f638772d3aa39372278b3d182d3053e849f7f5921a68f |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1784172322_ig.png` | 73035 | 0a3b00b07d2e4ac90d4f638772d3aa39372278b3d182d3053e849f7f5921a68f |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1784172348_ig.png` | 73035 | 0a3b00b07d2e4ac90d4f638772d3aa39372278b3d182d3053e849f7f5921a68f |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1784172348_shap.html` | 112695 | d8e002ab773ede011603809657e8957b3d4cfdcafa0c6e116067ed502c530fbe |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1784172348_shap.png` | 77390 | c6e9cfe9bb5a4011298e84162f59aa49f771d07d8a2d12d543384f8644ed00b1 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1784172720_ig.png` | 54766 | 3013656b5803b88ba13e4011e7f4806c10a80a81878fe2d7a6b37898b2300358 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1784172731_ig.png` | 54766 | 3013656b5803b88ba13e4011e7f4806c10a80a81878fe2d7a6b37898b2300358 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1784172731_shap.html` | 49500 | 674bb08a0c1ba5efdb7f280777c1727b1d72f444f9cac3f56de3a2ed12009144 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1784172731_shap.png` | 55099 | 53b025cf436fe2d3e428e288bd95dfd03f9fc579fafc3d0c662900e53003b7d8 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1784981036_integrated_hessians.png` | 100589 | f7b5800ac19846291ac3c197dab803468ee6f653d4bb147a10179752326b22e9 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1784981170_integrated_hessians.png` | 100589 | f7b5800ac19846291ac3c197dab803468ee6f653d4bb147a10179752326b22e9 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1785664390_ig.png` | 73035 | 0a3b00b07d2e4ac90d4f638772d3aa39372278b3d182d3053e849f7f5921a68f |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1785664390_shap.html` | 112695 | a1ce84113520623f6f819373913b72d6ad3e291a455c55fb859771524eea9805 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1785664390_shap.png` | 77390 | c6e9cfe9bb5a4011298e84162f59aa49f771d07d8a2d12d543384f8644ed00b1 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1785665314_lime.html` | 1232866 | 1e0d19e056375d9625ee3bcd167edfedb3eea1ef8065a0d1678951cc09dcd70e |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1785665314_lime.png` | 60999 | f3cd2cfef3216f4a97a191e73cdea2065f30a7bf4d6d2ce6e832d3dbe6d7a087 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1785665370_ig.png` | 73035 | 0a3b00b07d2e4ac90d4f638772d3aa39372278b3d182d3053e849f7f5921a68f |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1785665370_lime.html` | 1232866 | 1e0d19e056375d9625ee3bcd167edfedb3eea1ef8065a0d1678951cc09dcd70e |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1785665370_lime.png` | 60999 | f3cd2cfef3216f4a97a191e73cdea2065f30a7bf4d6d2ce6e832d3dbe6d7a087 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1785665370_shap.html` | 112695 | 2f0454f496e1768b7e0108c0497a23a8966c2342ea689e03a95d9a36573fa0b1 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1785665370_shap.png` | 77390 | c6e9cfe9bb5a4011298e84162f59aa49f771d07d8a2d12d543384f8644ed00b1 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1785665429_integrated_hessians.png` | 64684 | 80d98e656f6512eea031b943052b9f3d247414b9b74b574be6932dcdf5d3b773 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1785665765_ig.png` | 73035 | 0a3b00b07d2e4ac90d4f638772d3aa39372278b3d182d3053e849f7f5921a68f |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1785666169_combined_xai.png` | 64584 | 9dcec0c828b4bf269a3de00d61fd70fbcf5079b52c5650d85a14013a24c2470f |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1785686173_combined_xai.png` | 64584 | 9dcec0c828b4bf269a3de00d61fd70fbcf5079b52c5650d85a14013a24c2470f |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/deberta_v3_1785686760_combined_xai.png` | 64584 | 9dcec0c828b4bf269a3de00d61fd70fbcf5079b52c5650d85a14013a24c2470f |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/mentalbert_1779863965_ig.png` | 71442 | 46848c27f3ff7d3255f7fc0d976d832dba1c57bd2695e95860bb9895cc1cac77 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/mentalbert_1779863965_shap.html` | 112030 | a708fc09cdd466e7378886d31e8303f23af3ec69b21fded510037427dc20e2b9 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/mentalbert_1779863965_shap.png` | 75653 | a5f557306023cdcfd0e34f1a8c84976c72f52b047e283fd18c4ba9982fea8497 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/mentalbert_1779864191_ig.png` | 71442 | 46848c27f3ff7d3255f7fc0d976d832dba1c57bd2695e95860bb9895cc1cac77 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/mentalbert_1779864191_shap.html` | 112030 | 4cd6da1ad43b10b228a67027d019dffa75e63b615373ca742c6668f5b694c50f |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/mentalbert_1779864191_shap.png` | 75653 | a5f557306023cdcfd0e34f1a8c84976c72f52b047e283fd18c4ba9982fea8497 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/mentalbert_1779864335_shap.html` | 112030 | ef4a821788a096a0e171dd1635d18eb94a8e9c21b052aa4365794cfa8bfb76d5 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/mentalbert_1779864335_shap.png` | 75653 | a5f557306023cdcfd0e34f1a8c84976c72f52b047e283fd18c4ba9982fea8497 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/mentalbert_1779864738_ig.png` | 71442 | 46848c27f3ff7d3255f7fc0d976d832dba1c57bd2695e95860bb9895cc1cac77 |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/mentalbert_1779864738_shap.html` | 112030 | 52e8b49bcfd2cc10d5bcb95a8b7d1c490cff8b61aa596d43315f6ac6c3a2975e |
| `reports/c3_text_stressor_distortion/Stress header/evaluation/xai_outputs/streamlit/mentalbert_1779864738_shap.png` | 75653 | a5f557306023cdcfd0e34f1a8c84976c72f52b047e283fd18c4ba9982fea8497 |
| `reports/c3_text_stressor_distortion/Stress header/token_lengths_3models.png` | 65591 | a661d5f311787d958764913b099360c9da5af81458aeb00074b70619f64f21fe |


## Final deliverable review

CSV schema, source existence, numerical tables, privacy and write scope are checked after authoring. Review results are appended below. Conflicting experiments remain separate. No claim of systematic review, participant consent, clinical effectiveness, commercial clearance or individual authorship was added without evidence.

Final review completed: exactly the five requested files exist in the new directory. The claim CSV has 121 rows with all six required fields populated and source paths checked; 185 existing visual/table assets and 273 hashed artifact entries are recorded. Nineteen saved model-result comparisons (16 later/V4 plus3 S-BEST) match their own artifacts at1e-9; calibration comparison and probability errors are reported separately. Denominators, averaging conventions and conflicting experiment identities were reviewed against the source JSON/CSVs. Function/class names and source paths were corrected against current code during review. Privacy scans found no student-number strings, credential-token patterns or email addresses in the deliverables; source paths and scholarly author names are retained where appropriate, with the personal-name/student-number proposal filename withheld. No raw diary/patient/Reddit passage was reproduced.

A second streaming SHA-256 pass over all273 baseline artifacts (9,371,743,932bytes) completed in19.07seconds with zero changed bytes/hashes. Final `git diff --stat` is empty and `git status --short` shows only `?? docs/thesis_evidence/`. This establishes no tracked source changes and unchanged audited data/checkpoint/report files. No commit or push was made. Mock API startup/ASGI timeouts have no established cause and are not diagnosed as an application defect. The direct SQLite checks passed; API/browser/full-process-restart evidence remains missing.
