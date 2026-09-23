"""Install a reviewed overlay into a separate upstream checkout. Dry-run by default."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def main():
    repo = Path(__file__).resolve().parents[1]
    lock = json.loads((repo / 'dependencies.lock.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component', choices=list(lock), required=True)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    revision = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    expected = lock[args.component]['revision']
    if expected is None or revision != expected:
        parser.error(f'Unverified/mismatched base revision: expected {expected}, found {revision}. Resolve dependency lock first.')
    source = repo / 'overlays' / args.component
    files = sorted(p for p in source.rglob('*') if p.is_file())
    if not files:
        parser.error('No overlay files for this component')
    manifest = {r['path']: r['release_sha256'] for r in json.loads((repo / 'code_manifest.json').read_text())}
    operations = []
    for path in files:
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[str(path.relative_to(repo))]:
            parser.error(f'Overlay checksum mismatch: {path}')
        rel = path.relative_to(source)
        target = checkout / rel
        if checkout not in target.resolve().parents:
            parser.error(f'Unsafe target path: {target}')
        backup = checkout / '.roborecover-backup' / rel
        if target.exists() and target.read_bytes() == path.read_bytes():
            continue
        if backup.exists():
            parser.error(f'Backup already exists; resolve local edits first: {backup}')
        operations.append((path, target, backup))
    for path, target, backup in operations:
        print(f'{"Install" if args.apply else "Would install"}: {target}')
        if args.apply:
            if target.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    print('Done.' if args.apply else 'Dry-run only. Review, then repeat with --apply.')


if __name__ == '__main__':
    main()
