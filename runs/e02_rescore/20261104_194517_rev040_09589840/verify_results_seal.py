"""Read-only, standard-library verification of the completed rescore evidence seal."""
from pathlib import Path
import hashlib
import json
import os
import stat

ROOT = Path(__file__).absolute().parent
CONTROL_FILES = {'final_manifest.json', 'final_verification.json'}


def safe(path, leaf_link=False):
    assert path.is_absolute() and path.is_relative_to(ROOT)
    assert '..' not in path.parts
    for item in reversed(path.parents):
        assert not item.is_symlink(), str(item)
    if not leaf_link:
        assert not path.is_symlink(), str(path)
    return path


def metadata(s):
    return {'size': s.st_size, 'mode': s.st_mode, 'dev': s.st_dev,
            'ino': s.st_ino, 'mtime_ns': s.st_mtime_ns}


def regular_record(path):
    safe(path)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        before = os.fstat(stream.fileno())
        assert stat.S_ISREG(before.st_mode), str(path)
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
        after = os.fstat(stream.fileno())
    assert metadata(before) == metadata(after), str(path)
    return dict(metadata(after), sha256=digest.hexdigest())


def inventory():
    safe(ROOT)
    files, links = {}, {}
    for base, directories, names in os.walk(ROOT, followlinks=False):
        for name in list(directories) + names:
            path = Path(base) / name
            safe(path, leaf_link=True)
            s = path.lstat()
            relative = str(path.relative_to(ROOT))
            if stat.S_ISLNK(s.st_mode):
                links[relative] = dict(metadata(s), target=os.readlink(path))
                if name in directories:
                    directories.remove(name)
            elif stat.S_ISREG(s.st_mode):
                files[relative] = regular_record(path)
            else:
                assert stat.S_ISDIR(s.st_mode), str(path)
    return files, links


def verify():
    manifest_path = safe(ROOT / 'final_manifest.json')
    receipt_path = safe(ROOT / 'final_verification.json')
    manifest = json.loads(manifest_path.read_bytes())
    receipt = json.loads(receipt_path.read_bytes())
    assert manifest['root'] == str(ROOT)
    assert set(manifest['control_files']) == CONTROL_FILES
    assert receipt['manifest_record'] == regular_record(manifest_path)
    assert receipt['status'] == 'PASS'
    files, links = inventory()
    expected_names = set(manifest['files']) | CONTROL_FILES
    assert set(files) == expected_names, {
        'added': sorted(set(files) - expected_names),
        'missing': sorted(expected_names - set(files))}
    assert links == manifest['links'], 'Link target or identity changed'
    changed = [p for p, value in manifest['files'].items() if files[p] != value]
    assert not changed, changed
    assert receipt['total_regular_files_including_controls'] == len(files)
    assert receipt['total_links'] == len(links)
    return {'status': 'PASS', 'sealed_evidence_regular_files': len(manifest['files']),
            'total_regular_files_including_controls': len(files), 'total_links': len(links),
            'manifest_sha256': files['final_manifest.json']['sha256'],
            'final_verification_sha256': files['final_verification.json']['sha256'],
            'exact_set_and_metadata_unchanged': True, 'pure_read_only': True,
            'self_reference_rule': 'Both control paths are enumerated. The receipt hashes the manifest; its own hash is returned by this read-only invocation.'}


if __name__ == '__main__':
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
