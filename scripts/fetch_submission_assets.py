"""Fetch explicitly selected public assets and verify their original SHA-256."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fetch(asset):
    relative = Path(asset['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe asset path')
    target = ROOT / relative
    if any(parent.is_symlink() for parent in [target, *target.parents]):
        raise ValueError('Asset paths cannot contain symlinks')
    if target.exists():
        if file_hash(target) != asset['sha256']:
            raise ValueError('An existing asset has a different hash: ' + asset['path'])
        return 'already_verified'
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix='.asset-', dir=target.parent)
    temporary = Path(temporary_name)
    try:
        request = urllib.request.Request(asset['url'], headers={'User-Agent': 'AAE5103-coursework-asset-fetch/1.0'})
        with os.fdopen(descriptor, 'wb') as stream, urllib.request.urlopen(request, timeout=120) as response:
            while block := response.read(1024 * 1024):
                stream.write(block)
        if file_hash(temporary) != asset['sha256']:
            raise ValueError('Downloaded bytes differ from the frozen asset; nothing installed: ' + asset['path'])
        if target.exists():
            raise ValueError('Asset path appeared during download: ' + asset['path'])
        temporary.rename(target)
        return 'downloaded_and_verified'
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-data', action='store_true', help='Fetch the 16 original CWRU MAT records instead of the weight.')
    parser.add_argument('--historical-sources', action='store_true', help='Fetch original webpage snapshots for legacy audits; changed web bytes are rejected.')
    parser.add_argument('--check', action='store_true', help='Check local assets without downloading.')
    args = parser.parse_args()
    if args.raw_data and args.historical_sources:
        parser.error('Choose one asset group at a time.')
    group = 'raw_data' if args.raw_data else 'source_snapshots' if args.historical_sources else 'weights'
    assets = json.loads((ROOT / 'external_assets.json').read_bytes())[group]
    results = []
    for asset in assets:
        if args.check:
            target = ROOT / asset['path']
            state = 'missing' if not target.is_file() else 'verified' if file_hash(target) == asset['sha256'] else 'hash_mismatch'
        else:
            state = fetch(asset)
        results.append({'path': asset['path'], 'status': state})
    print(json.dumps({'group': group, 'assets': results}, indent=2))
    if args.check and any(row['status'] != 'verified' for row in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
