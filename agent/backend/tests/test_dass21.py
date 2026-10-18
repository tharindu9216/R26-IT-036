import tempfile
import unittest
from pathlib import Path

import dass21
from chat_store import ChatStore


class DASS21ScoringTests(unittest.TestCase):
    def test_scale_item_mapping_and_doubling(self):
        # Every item belongs to exactly one of the three scales.
        self.assertEqual(sorted(i for items in dass21.SCALE_ITEMS.values() for i in items), list(range(1, 22)))
        for scale, items in dass21.SCALE_ITEMS.items():
            with self.subTest(scale=scale):
                answers = {str(i): 3 if i in items else 0 for i in range(1, 22)}
                result = dass21.score_answers(answers)
                self.assertEqual(result[scale], {'raw_score': 21, 'score': 42, 'severity': 'Extremely Severe'})
                self.assertTrue(all(result[other]['score'] == 0 for other in result if other != scale))

    def test_normal_and_all_severity_boundaries(self):
        result = dass21.score_answers(dict.fromkeys((str(i) for i in range(1, 22)), 0))
        self.assertTrue(all(value['severity'] == 'Normal' for value in result.values()))
        for scale, boundaries in {'depression': (9, 13, 20, 27), 'anxiety': (7, 9, 14, 19), 'stress': (14, 18, 25, 33)}.items():
            for index, boundary in enumerate(boundaries):
                self.assertEqual(dass21.severity_for(scale, boundary), dass21.SEVERITIES[index])
                self.assertEqual(dass21.severity_for(scale, boundary + 1), dass21.SEVERITIES[index + 1])

    def test_incomplete_extra_and_invalid_answers_are_rejected(self):
        good = dict.fromkeys((str(i) for i in range(1, 22)), 0)
        for invalid in ({}, {**good, '22': 1}, {key: value for key, value in good.items() if key != '1'},
                        *({**good, '1': value} for value in (-1, 4, True, 1.5, '2'))):
            with self.subTest(answers=invalid), self.assertRaises(ValueError):
                dass21.score_answers(invalid)

    def test_history_survives_chat_reset_and_database_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'chat.sqlite3'
            store = ChatStore(path)
            answers = dict.fromkeys((str(i) for i in range(1, 22)), 2)
            record = store.save_dass21('patient-a', dass21.VERSION, answers, dass21.score_answers(answers))
            store.append('patient-a', 'user', 'hello')
            store.reset('patient-a')
            store.close()
            reopened = ChatStore(path)
            self.addCleanup(reopened.close)
            self.assertEqual(reopened.dass21_history('patient-a'), [record])
            self.assertEqual(reopened.dass21_history('patient-b'), [])
            self.assertEqual(reopened.history('patient-a'), [])
