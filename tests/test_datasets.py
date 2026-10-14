import json
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

from octopus.data import prepare_sample, format_sample, records, score_prediction


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=True):
        return ([1] if add_special_tokens else []) + [ord(c) for c in text]


class DatasetTests(unittest.TestCase):
    def test_mmlu_preserves_choices_and_excludes_answer(self):
        record = {'question': 'Which letter?', 'choices': ['one', 'two', 'three', 'four'], 'answer': 2}
        _, prompt, _, meta = format_sample(record, 'mmlu')
        self.assertIn('C. three', prompt)
        self.assertTrue(prompt.endswith('Answer:'))
        self.assertEqual(meta['reference'], 'C')
        self.assertEqual(score_prediction('C. three', meta)['value'], 1)
        self.assertEqual(score_prediction('A', meta)['value'], 0)

    def test_gsm8k_reference_is_never_in_prompt(self):
        _, prompt, _, meta = format_sample({'question': 'What is 2+3?', 'answer': 'Secret rationale. #### 5'}, 'gsm8k')
        self.assertNotIn('Secret', prompt)
        self.assertEqual(score_prediction('Calculation. #### 5.0', meta)['value'], 1)
        self.assertEqual(score_prediction('#### 6', meta)['value'], 0)

    def test_all_dataset_formats_and_question_preservation(self):
        samples = {
            'context': {'context': 'x' * 300},
            'mmlu': {'question': 'Pick one', 'choices': ['red', 'blue', 'black', 'white'], 'answer': 1},
            'gsm8k': {'question': 'Compute 2+2', 'answer': '#### 4'},
            'wikitext': {'text': 'x' * 300},
            'longbench': {'context': 'x' * 300, 'input': 'What is the key?', 'answers': ['answer secret']},
            'infinitebench': {'context': 'x' * 300, 'input': 'What is the key?', 'answer': 'answer secret'},
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, record in samples.items():
                with self.subTest(dataset=name):
                    path = Path(directory) / f'{name}.jsonl'
                    path.write_text(json.dumps(record) + '\n')
                    args = SimpleNamespace(dataset=name, dataset_path=path, dataset_config=None,
                                           dataset_split='test', sample_index=0, prefix_length=200)
                    ids, meta = prepare_sample(CharacterTokenizer(), args)
                    self.assertLessEqual(len(ids), 200)
                    text = ''.join(chr(t) for t in ids[1:])
                    self.assertNotIn('answer secret', text)
                    if name in ('longbench', 'infinitebench'):
                        self.assertIn(record['input'], text)
                        self.assertTrue(meta['truncated'])

    def test_short_questions_are_not_padded_and_long_questions_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.jsonl'
            path.write_text(json.dumps({'question': 'Why?', 'answer': '#### 42'}) + '\n')
            args = SimpleNamespace(dataset='gsm8k', dataset_path=path, dataset_config=None,
                                   dataset_split='test', sample_index=0, prefix_length=1024)
            ids, meta = prepare_sample(CharacterTokenizer(), args)
            self.assertLess(len(ids), 1024)
            self.assertFalse(meta['truncated'])
            args.prefix_length = 8
            with self.assertRaisesRegex(ValueError, 'Complete question'):
                prepare_sample(CharacterTokenizer(), args)

    def test_wikitext_joins_rows_without_repeating_and_selects_index(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.jsonl'
            path.write_text('\n'.join(json.dumps({'text': x}) for x in ('', 'skip', 'alpha', 'beta', 'gamma')))
            args = SimpleNamespace(dataset='wikitext', dataset_path=path, dataset_config=None,
                                   dataset_split='test', sample_index=1, prefix_length=12)
            ids, _ = prepare_sample(CharacterTokenizer(), args)
            self.assertEqual(''.join(chr(x) for x in ids[1:]), 'alpha\n\nbeta')
            args.prefix_length = 100
            with self.assertRaisesRegex(ValueError, 'Input contains'):
                prepare_sample(CharacterTokenizer(), args)

    def test_local_json_and_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'a.json').write_text(json.dumps({'test': [{'text': 'a'}]}))
            self.assertEqual(list(records(root / 'a.json', 'wikitext')), [{'text': 'a'}])
            (root / 'a.txt').write_text('a\n\nb\n')
            self.assertEqual(list(records(root / 'a.txt', 'wikitext')), [{'text': 'a'}, {'text': 'b'}])

    def test_long_context_task_defaults_and_invalid_mmlu(self):
        for task in ('gov_report', 'longbook_sum_eng', 'passkey', 'number_string', 'math_calc', 'longdialogue_qa_eng'):
            _, _, suffix, _ = format_sample({'context': 'input document'}, 'infinitebench', task)
            self.assertTrue(suffix.endswith('Answer:'))
        for answer in (4, [], 'AB'):
            with self.assertRaises(ValueError):
                format_sample({'question': 'Q', 'choices': ['a','b','c','d'], 'answer': answer}, 'mmlu')

    def test_parquet_and_saved_dataset_inputs(self):
        from datasets import Dataset, DatasetDict
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = Dataset.from_dict({'text': ['one', 'two']})
            data.to_parquet(root / 'input.parquet')
            data.save_to_disk(str(root / 'single'))
            DatasetDict(test=data).save_to_disk(str(root / 'splits'))
            for path in (root / 'input.parquet', root / 'single', root / 'splits'):
                self.assertEqual(list(records(path, 'wikitext')), [{'text': 'one'}, {'text': 'two'}])

    def test_launcher_hub_selection_and_required_task(self):
        from octopus.main import make_parser, plan
        parser = make_parser()
        base = ['--model-path', '/model', '--work-dir', '/run', '--dram-only']
        args = parser.parse_args(base + ['--dataset', 'mmlu'])
        commands, _, dataset, _, _ = plan(args, parser)
        self.assertIsNone(dataset)
        self.assertIn('mmlu', commands[0])
        args = parser.parse_args(base + ['--dataset', 'longbench'])
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            plan(args, parser)
        args = parser.parse_args(base + ['--dataset', 'longbench', '--dataset-config', 'qasper'])
        self.assertIn('qasper', plan(args, parser)[0][0])
        with self.assertRaises(ValueError):
            next(records(None, 'infinitebench', '../invalid'))


if __name__ == '__main__':
    unittest.main()
