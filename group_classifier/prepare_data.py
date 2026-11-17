"""Restore the collaborator's complete dataset from the checked-in archive volumes."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parent


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def target_path(name):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts or relative.parts[0] != 'Data':
        raise ValueError('Unsafe dataset path: ' + name)
    target = ROOT / relative
    if any(p.is_symlink() for p in [target, *target.parents]):
        raise ValueError('Dataset paths cannot contain symbolic links: ' + name)
    return target


def check_files(files):
    missing = []
    for name, record in files.items():
        target = target_path(name)
        if not target.exists():
            missing.append(name)
        elif not target.is_file() or digest(target) != record['sha256']:
            raise ValueError('An existing dataset file differs; it was not overwritten: ' + name)
    return missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true', help='Check extracted files without extracting.')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'dataset/manifest.json').read_bytes())
    files = {r['path']: r for r in manifest['files']}
    missing = check_files(files)
    if args.verify_only and missing:
        raise SystemExit(f'{len(missing)} dataset files are missing. Run without --verify-only to restore them.')
    restored = 0
    if missing:
        with tempfile.TemporaryFile() as assembled:
            archive_digest = hashlib.sha256()
            for volume in manifest['volumes']:
                path = ROOT / 'dataset' / volume['name']
                if path.parent != ROOT / 'dataset' or path.is_symlink() or digest(path) != volume['sha256']:
                    raise ValueError('Archive volume differs: ' + volume['name'])
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        archive_digest.update(block)
                        assembled.write(block)
            if archive_digest.hexdigest() != manifest['archive_sha256']:
                raise ValueError('Combined dataset archive differs from its recorded hash')
            assembled.seek(0)
            seen = set()
            with tarfile.open(fileobj=assembled, mode='r:gz') as archive:
                for member in archive:
                    if not member.isfile() or member.name not in files or member.name in seen:
                        raise ValueError('Unexpected dataset archive member: ' + member.name)
                    seen.add(member.name)
                    target = target_path(member.name)
                    if target.exists():
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    fd, temporary_name = tempfile.mkstemp(prefix='.restore-', dir=target.parent)
                    temporary = Path(temporary_name)
                    try:
                        with os.fdopen(fd, 'wb') as output, archive.extractfile(member) as stream:
                            shutil.copyfileobj(stream, output)
                        if digest(temporary) != files[member.name]['sha256']:
                            raise ValueError('Extracted dataset bytes differ: ' + member.name)
                        os.link(temporary, target)
                        restored += 1
                    finally:
                        temporary.unlink(missing_ok=True)
            if seen != set(files):
                raise ValueError('Dataset archive is incomplete')
        if check_files(files):
            raise ValueError('Dataset restoration is incomplete')
    print(json.dumps({'status': 'PASS', 'verified_files': len(files), 'restored_files': restored,
                      'original_dataset_bytes': sum(r['bytes'] for r in files.values()),
                      'network_requests': 0, 'training_performed': False}, indent=2))


if __name__ == '__main__':
    main()
