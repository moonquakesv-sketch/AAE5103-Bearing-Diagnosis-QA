"""Small shared primitives for the isolated E02 runtime, with no import-time I/O."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import stat


class Rejected(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise Rejected(reason)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def snapshot(value):
    return strict_json(canonical(value))


def strict_json(value):
    def pairs(items):
        obj = {}
        for k, v in items:
            require(k not in obj, 'duplicate JSON key')
            obj[k] = v
        return obj
    def invalid(_):
        raise Rejected('non-finite JSON')
    return json.loads(value, object_pairs_hook=pairs, parse_constant=invalid)


def safe_path(path, root=None):
    path = Path(path).absolute()
    require('..' not in path.parts, 'parent traversal')
    for p in [*reversed(path.parents), path]:
        if p.is_symlink():
            raise Rejected('symbolic path is not permitted')
    if root is not None:
        require(path.is_relative_to(Path(root).absolute()), 'path outside permitted root')
    return path


def read_bytes(path):
    p = safe_path(path)
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as f:
        require(stat.S_ISREG(os.fstat(f.fileno()).st_mode), 'not an ordinary file')
        return f.read()


def file_hash(path):
    return hashlib.sha256(read_bytes(path)).hexdigest()


def read_json(path):
    return strict_json(read_bytes(path))


def write_new(path, value, root):
    p = safe_path(path, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = (canonical(value) + '\n').encode('utf-8')
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    return hashlib.sha256(raw).hexdigest()


BUSINESS_FIELDS = {'predicted_class', 'scenario_id', 'question', 'task_types', 'explicitly_allowed_operating_context'}
CLASSES = ['normal', 'inner_race', 'outer_race', 'rolling_element']
TASKS = ['explanation', 'inspection', 'preparation']


def validate_business(request, scenarios, allowed_contexts=None):
    require(type(request) is dict and set(request) == BUSINESS_FIELDS, 'five business fields required')
    require(request['predicted_class'] in CLASSES, 'invalid class')
    require(request['scenario_id'] in scenarios, 'unknown scenario')
    require(scenarios[request['scenario_id']]['predicted_class'] == request['predicted_class'], 'class/scenario conflict')
    require(type(request['question']) is str and request['question'].strip(), 'empty question')
    request['question'].encode('utf-8')
    tasks = request['task_types']
    require(type(tasks) is list and tasks and all(type(t) is str for t in tasks), 'invalid tasks')
    require(len(tasks) == len(set(tasks)) and set(tasks) <= set(TASKS), 'unknown/duplicate task')
    require(type(request['explicitly_allowed_operating_context']) is dict, 'context must be object')
    canonical(request).encode('utf-8')
    if allowed_contexts is not None:
        require(request['explicitly_allowed_operating_context'] == allowed_contexts[request['scenario_id']], 'unapproved common context')
