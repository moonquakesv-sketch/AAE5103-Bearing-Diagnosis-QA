"""Read-only standard-library verifier for this preparation and its source-history baseline."""
from pathlib import Path
import hashlib
import json
import os
import stat

ROOT = Path(__file__).absolute().parent
PROJECT = ROOT.parents[2]


def guard(path):
    assert '..' not in path.parts and path.is_relative_to(PROJECT)
    for p in [*reversed(path.parents), path]:
        assert not p.is_symlink(), str(p)
    return path


def sha(path):
    h = hashlib.sha256()
    with guard(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(guard(path).read_bytes())


def check():
    binding = read(ROOT / 'config/execution_binding.json')
    for relative, expected in binding['files'].items():
        assert not Path(relative).is_absolute()
        assert sha(PROJECT / relative) == expected, relative
    baseline = read(ROOT / 'baseline/workspace_start.json')
    exclusions = {ROOT} | {PROJECT / p for p in baseline['exact_active_demo_exclusions']}
    files, links = {}, {}
    for base, directories, names in os.walk(PROJECT, followlinks=False):
        for name in list(directories) + names:
            path = Path(base) / name
            if path in exclusions:
                if name in directories:
                    directories.remove(name)
                continue
            s = path.lstat()
            relative = str(path.relative_to(PROJECT))
            value = {'size': s.st_size, 'mode': s.st_mode, 'dev': s.st_dev, 'ino': s.st_ino, 'mtime_ns': s.st_mtime_ns}
            if stat.S_ISLNK(s.st_mode):
                links[relative] = dict(value, target=os.readlink(path))
                if name in directories:
                    directories.remove(name)
            elif stat.S_ISREG(s.st_mode):
                files[relative] = dict(value, sha256=sha(path))
    assert files == baseline['files'], {'missing': sorted(set(baseline['files']) - set(files)),
                                       'added': sorted(set(files) - set(baseline['files'])),
                                       'changed': [p for p in files if p in baseline['files'] and files[p] != baseline['files'][p]]}
    assert links == baseline['links']
    copies = read(ROOT / 'inputs/source_copies.json')
    for row in copies:
        assert sha(PROJECT / row['source']) == row['sha256'] == sha(ROOT / row['copy'])
    return {'status': 'PASS', 'bound_files': len(binding['files']), 'preserved_regular_files': len(files),
            'preserved_links': len(links), 'unchanged_source_copies': len(copies),
            'binding_sha256': sha(ROOT / 'config/execution_binding.json'), 'pure_read_only': True}


if __name__ == '__main__':
    print(json.dumps(check(), ensure_ascii=False, indent=2))
