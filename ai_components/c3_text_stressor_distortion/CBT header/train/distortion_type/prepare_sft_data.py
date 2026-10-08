"""Prepare leakage-safe SFT and evaluation JSONL for distortion-type SLMs.

Only labels 1-10 are included. The existing real train/validation/test split
is preserved exactly. Test references are written in an evaluation-only
schema rather than an SFT messages schema to reduce accidental test training.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from config import TypeSLMConfig


LABELS = {
    1: 'All-or-nothing thinking',
    2: 'Overgeneralization',
    3: 'Mental filter',
    4: 'Should statements',
    5: 'Labeling',
    6: 'Personalization',
    7: 'Magnification',
    8: 'Emotional Reasoning',
    9: 'Mind Reading',
    10: 'Fortune-telling',
}

DEFINITIONS = {
    1: 'Viewing situations in absolute, black-or-white categories.',
    2: 'Drawing a broad negative conclusion from limited events.',
    3: 'Focusing on negative details while filtering out relevant positives.',
    4: 'Using rigid should, must, or ought expectations.',
    5: 'Assigning a fixed negative identity label to oneself or others.',
    6: 'Assuming disproportionate personal responsibility for an outcome.',
    7: 'Exaggerating the importance, severity, or consequences of an event.',
    8: 'Treating a feeling as proof that something is objectively true.',
    9: 'Assuming another person’s thoughts without sufficient evidence.',
    10: 'Predicting a negative future outcome without sufficient evidence.',
}

SYSTEM_PROMPT = (
    'You are a research classifier for cognitive distortions. '
    'Choose exactly one dominant type from the supplied taxonomy. '
    'Use only evidence present in the patient text. Return valid JSON with '
    'label_id, label, evidence, and explanation. This is classification, '
    'not a clinical diagnosis or treatment recommendation.'
)


def taxonomy_text():
    return '\n'.join(
        f'{label_id}. {LABELS[label_id]}: {DEFINITIONS[label_id]}'
        for label_id in LABELS
    )


def user_prompt(text):
    return (
        f'CBT distortion taxonomy:\n{taxonomy_text()}\n\n'
        f'Patient text:\n{text}\n\n'
        'Identify the single dominant distortion type.'
    )


def reference_output(row):
    label_id = int(row['label'])
    raw_evidence = row.get('Distorted part', '')
    evidence = (
        ''
        if pd.isna(raw_evidence)
        else str(raw_evidence).strip()
    )
    return {
        'label_id': label_id,
        'label': LABELS[label_id],
        'evidence': evidence,
        'explanation': (
            f'The evidence is most consistent with '
            f'{LABELS[label_id]}.'
        ),
    }


def sft_record(row):
    return {
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {
                'role': 'user',
                'content': user_prompt(str(row['Patient Question'])),
            },
            {
                'role': 'assistant',
                'content': json.dumps(
                    reference_output(row), ensure_ascii=False),
            },
        ],
        'source_id': str(row.get('Id_Number', '')),
    }


def evaluation_record(row):
    return {
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {
                'role': 'user',
                'content': user_prompt(str(row['Patient Question'])),
            },
        ],
        'reference': reference_output(row),
        'source_id': str(row.get('Id_Number', '')),
    }


def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False) + '\n')


def prepare(data_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}

    for split in ('train', 'val', 'test'):
        frame = pd.read_csv(data_dir / f'cbt_{split}.csv')
        frame = frame[frame['label'].isin(LABELS)].copy()
        if split == 'test':
            records = [
                evaluation_record(row)
                for _, row in frame.iterrows()
            ]
            filename = 'type_test_evaluation.jsonl'
        else:
            records = [
                sft_record(row)
                for _, row in frame.iterrows()
            ]
            filename = f'type_{split}_sft.jsonl'
        write_jsonl(records, output_dir / filename)
        counts[split] = {
            'rows': len(records),
            'label_counts': {
                str(int(key)): int(value)
                for key, value
                in frame['label'].value_counts().sort_index().items()
            },
        }

    with (output_dir / 'dataset_manifest.json').open(
            'w', encoding='utf-8') as handle:
        json.dump(
            {
                'task': 'conditional_10_class_distortion_type',
                'test_is_evaluation_only': True,
                'counts': counts,
                'labels': LABELS,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=TypeSLMConfig.DATA_DIR,
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(__file__).resolve().parent / 'prepared_data',
    )
    arguments = parser.parse_args()
    counts = prepare(arguments.data_dir, arguments.output_dir)
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()
