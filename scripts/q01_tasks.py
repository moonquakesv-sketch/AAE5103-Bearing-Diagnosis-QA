"""Conservative English task routing; no model, retrieval, key or corpus reads.

This optional input adapter accepts four caller fields and adds task_types only
when every explicit clause resolves. It never replaces explicit experiment tags.
The prefix grammar is a narrow baseline, not general language understanding.
"""
import argparse
import os
from pathlib import Path
import re
import stat
import sys

import q01_flow as flow

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'q01.tasks.rules.v1'
INPUT_FIELDS = flow.k02.INPUT_FIELDS - {'task_types'}
DOMAIN = re.compile(r'\b(?:bearing|raceway|race|cage|vibration|fault|damage|maintenance)\b|\bthis (?:result|label)\b', re.I)
CONTROL = re.compile(r'\b(?:ignore|disregard|bypass|rubric|password|credentials)\b|\b(?:system prompt|reference answer|api key)\b', re.I)
NEGATED = re.compile(r"\b(?:not|never|without|instead|rather than|or|don\x27t)\b", re.I)
SPLIT = re.compile(r'[?;\n]+|\.\s+|\s*,?\s+(?:and(?:\s+also)?|also)\s+', re.I)
POLITE = re.compile(r'^(?:(?:please|also)\s+|(?:can|could|would)\s+you\s+)', re.I)
UNSUPPORTED = re.compile(r'^(?:write|calculate|compute|translate|predict|classify|diagnose|approve|certify|release)\b', re.I)
RULES = (
    ('explanation', 'explain_meaning', r'^(?:explain|describe)\s+(?:(?:the|this|that)\s+)?(?:meaning|cause|mechanism|result|label|fault|damage|vibration|condition)\b.+'),
    ('explanation', 'why', r'^why\b.+'),
    ('explanation', 'what_means', r'^what (?:does .+ mean|causes .+|is the (?:meaning|cause|mechanism).+)'),
    ('inspection', 'inspect', r'^(?:inspect|check|examine)\s+.+'),
    ('inspection', 'inspection_question', r'^(?:what|how) (?:should|can|must) (?:i |we )?(?:inspect|check|examine)\b.*'),
    ('inspection', 'passive_inspection', r'^what (?:should|can|must) be (?:checked|inspected|examined)\b.*'),
    ('inspection', 'inspection_steps', r'^(?:describe|list|explain) (?:the )?(?:inspection|checks|inspection steps|what to inspect)\b.*'),
    ('preparation', 'prepare', r'^prepare\s+.+'),
    ('preparation', 'prepare_question', r'^(?:what|how) (?:should|do|can|must) (?:i |we )?(?:prepare|gather|arrange)\b.*'),
    ('preparation', 'preparation_steps', r'^(?:list|describe|explain) (?:the )?(?:preparations|maintenance preparation|preparation steps)\b.*'),
)
COMPILED = tuple((task, name, re.compile(pattern, re.I)) for task, name, pattern in RULES)


def _record(status, code, message, tasks=None, matches=None, request=None):
    return {'version': VERSION, 'method': 'conservative_english_prefix_rules',
            'status': status, 'reason_code': code, 'service_message': message,
            'task_types': tasks or [], 'matches': matches or [], 'request': request,
            'confidence': 'not_calibrated', 'semantic_understanding_proven': False,
            'code_sha256': flow.file_hash(__file__)}


def recognize_tasks(question):
    """Return canonical tasks only for fully resolved, unquoted positive clauses."""
    if type(question) is not str or not question.strip() or len(question) > 2000:
        return _record('invalid_input', 'question_shape', 'Provide a nonblank English question of at most 2000 characters.')
    try:
        question.encode('utf-8')
    except UnicodeEncodeError:
        return _record('invalid_input', 'invalid_unicode', 'The question contains invalid Unicode.')
    if not question.isascii():
        return _record('unsupported', 'language_profile', 'This baseline accepts ASCII English; please restate the question.')
    if CONTROL.search(question):
        return _record('unsupported', 'control_or_evaluation_text', 'Restate the bearing task without control or evaluation instructions.')
    if any(c in question for c in ('"', "'", '`')) or NEGATED.search(question):
        return _record('needs_clarification', 'quoted_negated_or_alternative', 'Please state the positive tasks explicitly: explanation, inspection or preparation.')
    if not DOMAIN.search(question):
        return _record('unsupported', 'domain_not_explicit', 'Please ask an explicit bearing task or refer to this result or label.')
    clauses = [s.strip(' ,.!:\t\r') for s in SPLIT.split(question) if s.strip(' ,.!:\t\r')]
    if not clauses or len(clauses) > 6:
        return _record('needs_clarification', 'clause_count', 'Please split the request into at most six explicit task clauses.')
    tasks, matches, unknown, unsupported = set(), [], False, False
    for index, clause in enumerate(clauses):
        while POLITE.match(clause):
            clause = POLITE.sub('', clause, count=1)
        hits = [(task, name) for task, name, pattern in COMPILED if pattern.fullmatch(clause)]
        kinds = {task for task, name in hits}
        if UNSUPPORTED.match(clause) or not DOMAIN.search(clause):
            unsupported = True
        if len(kinds) != 1:
            unknown = True
        else:
            tasks.update(kinds)
            matches.append({'clause_index': index, 'rule_ids': [name for task, name in hits], 'task_type': next(iter(kinds))})
    if unsupported:
        return _record('unsupported', 'unsupported_task_clause', 'Part of the request is outside explanation, inspection and preparation.', matches=matches)
    if unknown:
        return _record('needs_clarification', 'unresolved_clause', 'Please identify every requested task explicitly; no partial task set has been dispatched.', matches=matches)
    ordered = [task for task in flow.k02.TASKS if task in tasks]
    return _record('ready', 'explicit_prefixes', None, ordered, matches)


def prepare_request(payload):
    """Copy four business fields; never accept/overwrite caller task_types."""
    try:
        value = flow.json_snapshot(payload)
    except Exception:
        return _record('invalid_input', 'unsafe_json', 'Input must contain only safe native JSON values.')
    if type(value) is not dict or set(value) != INPUT_FIELDS:
        return _record('invalid_input', 'input_fields', 'Provide exactly predicted_class, scenario_id, question and explicitly_allowed_operating_context. Use the existing flow directly for explicit task_types.')
    route = recognize_tasks(value['question'])
    if route['status'] != 'ready':
        return route
    request = dict(value, task_types=list(route['task_types']))
    try:
        flow.k02.validate_request(request)
    except Exception as error:
        return _record('invalid_input', getattr(error, 'code', 'invalid_request'), 'The business context does not satisfy the existing request contract.')
    route['request'] = flow.json_snapshot(request)
    return route


def route_and_run(payload, config, provider, *, retriever=None, runtime=None):
    """Dispatch at most once; a blocked route does not touch any engine object."""
    route = prepare_request(payload)
    record = None
    if route['status'] == 'ready':
        record = flow.run_flow(route['request'], config, provider, retriever=retriever, runtime=runtime)
    return {'routing': route, 'flow_record': record}


def _controlled(path):
    path = Path(path).absolute()
    if '..' in path.parts or not path.is_relative_to(ROOT / 'runs/q01_task'):
        raise ValueError('Development CLI files must remain under runs/q01_task.')
    for p in (path,) + tuple(path.parents):
        try:
            if stat.S_ISLNK(p.lstat().st_mode):
                raise ValueError('Symbolic links are not allowed.')
        except FileNotFoundError:
            continue
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        source, target = _controlled(args.input), _controlled(args.output)
        # Reserve before reading; an existing output is never overwritten.
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'wb') as output:
            try:
                if not stat.S_ISREG(source.lstat().st_mode):
                    raise ValueError('Input must be a regular file.')
                source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
                with os.fdopen(source_fd, 'rb') as handle:
                    raw = handle.read(65537)
                if len(raw) > 65536:
                    raise ValueError('Development input exceeds 64 KiB.')
                result = prepare_request(flow.k02.strict_json(raw))
            except Exception:
                result = _record('invalid_input', 'invalid_file_or_json', 'The input file is not a valid bounded JSON request.')
            output.write((flow.canonical(result) + '\n').encode('ascii'))
        print(flow.canonical({'status': result['status'], 'output': str(target)}))
        return 0 if result['status'] == 'ready' else 1
    except Exception as error:
        print(type(error).__name__ + ': ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
