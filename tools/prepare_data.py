"""Validate a downloaded scenario tree and produce evaluator input lists (stdlib)."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def prepare(platform, split, data_root, output):
    repository = Path(__file__).resolve().parents[1]
    with (repository / 'splits' / f'{platform}_{split}.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    expected = 800 if split == 'train' else 200
    if len(rows) != expected or len({r['sample_id'] for r in rows}) != expected:
        raise ValueError('Invalid split membership')
    paths = []
    for row in rows:
        path = (data_root / row['path']).resolve()
        if data_root.resolve() not in path.parents:
            raise ValueError('Scenario path escapes data root')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError(f'Checksum mismatch: {path}')
        data = json.loads(raw)
        actions = data['actions']
        if len(actions) != int(row['actions']) or not actions:
            raise ValueError(f'Invalid actions: {path}')
        widths = {len(a) for a in actions}
        if len(widths) != 1 or not widths <= ({7} if platform == 'libero' else {14, 16}):
            raise ValueError(f'Invalid action dimensions: {path}')
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for a in actions for x in a):
            raise ValueError(f'Nonfinite/non-numeric action: {path}')
        paths.append(str(path))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(paths) + '\n')
    print(f'Validated {len(paths)} {platform}/{split} scenarios; wrote {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platform', choices=['libero', 'robotwin'], required=True)
    parser.add_argument('--split', choices=['train', 'test'], default='test')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.platform, args.split, args.data_root, args.output)
