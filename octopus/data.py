import itertools
import json
import re
import zipfile
from pathlib import Path


DATASETS = ('context', 'mmlu', 'gsm8k', 'wikitext', 'longbench', 'infinitebench')
HUB_DATASETS = {'mmlu': ('cais/mmlu', 'all'), 'gsm8k': ('openai/gsm8k', 'main'),
                'wikitext': ('Salesforce/wikitext', 'wikitext-2-raw-v1')}


def add_dataset_arguments(parser, *, internal=False):
    sep = '_' if internal else '-'
    parser.add_argument('--dataset', choices=DATASETS, default='context')
    parser.add_argument('--dataset' + sep + 'config', help='MMLU subject, WikiText variant, or long-context task')
    parser.add_argument('--dataset' + sep + 'split', default='test')
    parser.add_argument('--sample' + sep + 'index', type=int, default=0)


def records(path, name, config=None, split='test'):
    if path is not None:
        path = Path(path)
        if path.suffix in ('.jsonl', '.txt'):
            with path.open(encoding='utf-8') as stream:
                for line in stream:
                    if line.strip():
                        yield {'text': line.rstrip()} if path.suffix == '.txt' else json.loads(line)
            return
        if path.suffix == '.json':
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                data = data.get(split, data.get('data', [data]))
            if not isinstance(data, list):
                raise ValueError('JSON input must contain a list of records')
            yield from data
            return
    if path is None and name in ('longbench', 'infinitebench'):
        if not config or not re.fullmatch(r'[A-Za-z0-9_-]+', config) or split != 'test':
            raise ValueError('Long-context Hub datasets require --dataset-config TASK and --dataset-split test')
        if name == 'longbench':
            from huggingface_hub import hf_hub_download
            archive = hf_hub_download('zai-org/LongBench', 'data.zip', repo_type='dataset')
            with zipfile.ZipFile(archive) as data:
                member = f'data/{config}.jsonl'
                if member not in data.namelist():
                    raise ValueError(f'Unknown LongBench task: {config}')
                with data.open(member) as stream:
                    for line in stream:
                        if line.strip():
                            yield json.loads(line)
            return
    from datasets import load_dataset, load_from_disk
    if path is not None:
        if path.is_dir():
            data = load_from_disk(str(path))
            yield from data[split] if hasattr(data, 'keys') else data
        elif path.suffix == '.parquet':
            yield from load_dataset('parquet', data_files=str(path), split='train', streaming=True)
        else:
            raise ValueError('Use JSONL, JSON, text, Parquet, or a datasets.save_to_disk directory')
    elif name in HUB_DATASETS:
        repo, default = HUB_DATASETS[name]
        yield from load_dataset(repo, config or default, split=split, streaming=True)
    elif name == 'infinitebench':
        url = f'https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/resolve/main/{config}.jsonl'
        yield from load_dataset('json', data_files=url, split='train', streaming=True)
    else:
        raise ValueError(f'--dataset-path is required for {name}; supply its local JSONL export')


def nonempty_text(record, key):
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'Dataset record needs a nonempty {key!r} string')
    return value


def format_sample(record, name, task=None):
    metadata = {'dataset': name, 'task': task, 'reference': None}
    if name in ('context', 'wikitext'):
        return '', nonempty_text(record, 'context' if name == 'context' else 'text'), '', metadata
    if name == 'mmlu':
        question = nonempty_text(record, 'question')
        choices = record.get('choices')
        if not isinstance(choices, (list, tuple)) or len(choices) != 4 or not all(isinstance(x, str) for x in choices):
            raise ValueError('MMLU requires four choices')
        answer = record.get('answer')
        if answer is not None:
            if isinstance(answer, int) and 0 <= answer < 4:
                answer = 'ABCD'[answer]
            if not isinstance(answer, str) or len(answer) != 1 or answer not in 'ABCD':
                raise ValueError('MMLU answer must be an index 0..3 or a letter A..D')
        metadata.update(reference=answer, subject=record.get('subject'))
        options = '\n'.join(f'{letter}. {choice}' for letter, choice in zip('ABCD', choices))
        return '', f'{question}\n{options}\nAnswer with one letter (A, B, C, or D).\nAnswer:', '', metadata
    if name == 'gsm8k':
        metadata['reference'] = record.get('answer')
        return '', f"Question: {nonempty_text(record, 'question')}\nSolve the problem and end with #### followed by the numeric answer.\nAnswer:", '', metadata
    context = nonempty_text(record, 'context')
    question = record.get('input', '')
    if not isinstance(question, str):
        raise ValueError('Long-context input must be a string')
    metadata['reference'] = record.get('answers', record.get('answer'))
    metadata['task'] = task or record.get('dataset', record.get('task'))
    options = record.get('options')
    if options:
        if not isinstance(options, list) or len(options) > 26:
            raise ValueError('Expected an options list with at most 26 choices')
        question += '\n' + '\n'.join(f'{chr(65+i)}. {text}' for i, text in enumerate(options))
    task = metadata['task'] or ''
    if not question.strip():
        if task in ('gov_report', 'qmsum', 'multi_news', 'vcsum', 'longbook_sum_eng'):
            question = 'Summarize the document.'
        elif task in ('lcc', 'repobench-p'):
            question = 'Continue the code with the next line.'
        elif task in ('passkey', 'number_string'):
            question = 'Return only the hidden pass key or number from the context.'
        elif task == 'longdialogue_qa_eng':
            question = 'Identify the speaker whose name was replaced by a placeholder.'
        elif task == 'math_calc':
            question = 'Evaluate the expression step by step and return the intermediate results.'
        else:
            raise ValueError('This task needs an input question or an explicit --dataset-config')
    return 'Read the following context.\n\n', context, f'\n\n{question}\nAnswer:', metadata


def prepare_sample(tokenizer, args):
    if args.sample_index < 0:
        raise ValueError('sample-index must be nonnegative')
    source = records(args.dataset_path, args.dataset, args.dataset_config, args.dataset_split)
    # Blank WikiText rows are separators, not independent evaluation examples.
    if args.dataset in ('context', 'wikitext'):
        key = 'context' if args.dataset == 'context' else 'text'
        source = (r for r in source if isinstance(r.get(key), str) and r[key].strip())
    source = itertools.islice(source, args.sample_index, None)
    record = next(source, None)
    if record is None:
        raise ValueError('sample-index is past the end of the dataset')
    start, body, end, metadata = format_sample(record, args.dataset, args.dataset_config)
    if args.dataset == 'wikitext':
        # Join consecutive text rows without repeating data or injecting answers.
        while len(tokenizer.encode(body)) < args.prefix_length:
            more = next(source, None)
            if more is None:
                break
            body += '\n\n' + more['text']
    ids = tokenizer.encode(start + body + end)
    original_tokens = len(ids)
    if args.dataset in ('context', 'wikitext'):
        if len(ids) < args.prefix_length:
            raise ValueError(f'Input contains {len(ids)} tokens; {args.prefix_length} are required')
        ids = ids[:args.prefix_length]
    elif args.dataset in ('mmlu', 'gsm8k'):
        if len(ids) > args.prefix_length:
            raise ValueError(f'Complete question requires {len(ids)} tokens; increase --prefix-length')
    elif len(ids) > args.prefix_length:
        prefix = tokenizer.encode(start)
        suffix = tokenizer.encode(end, add_special_tokens=False)
        budget = args.prefix_length - len(prefix) - len(suffix)
        if budget < 1:
            raise ValueError('prefix-length is too small to preserve the question and prompt')
        context = tokenizer.encode(body, add_special_tokens=False)
        front = (budget + 1) // 2
        back = budget - front
        ids = prefix + context[:front] + (context[-back:] if back else []) + suffix
    if not ids:
        raise ValueError('Tokenized input is empty')
    metadata.update(sample_index=args.sample_index, split=args.dataset_split,
                    original_tokens=original_tokens, prefix_tokens=len(ids), truncated=len(ids) < original_tokens)
    return ids, metadata


def score_prediction(text, metadata):
    reference = metadata.get('reference')
    if reference is None:
        return None
    name = metadata.get('dataset')
    if name == 'mmlu':
        match = re.search(r'^\s*\(?([A-D])\b', text)
        return {'metric': 'generated_choice_accuracy', 'value': float(bool(match) and match[1] == reference)}
    if name == 'gsm8k':
        def number(value):
            part = str(value).split('####')[-1].replace(',', '')
            values = re.findall(r'-?\d+(?:\.\d+)?', part)
            return values[-1] if values else None
        from decimal import Decimal
        actual, expected = number(text), number(reference)
        return {'metric': 'numeric_exact_match', 'value': float(actual is not None and expected is not None
                                                              and Decimal(actual) == Decimal(expected))}
    return None
