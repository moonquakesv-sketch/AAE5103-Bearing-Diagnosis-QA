"""Pinned official V4.1 text-chat counter; no network or credential access.

Counts the complete system/user prompt for the registered non-thinking JSON
profile. Known ambiguous literal special tokens are rejected, never rewritten.
An independent certificate is required before this counter can pass real gates.
"""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'deepseek-flash'


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('ascii')).hexdigest()


def file_hash(path):
    path = Path(path).absolute()
    for part in (path,) + tuple(path.parents):
        if part.exists() and stat.S_ISLNK(part.lstat().st_mode):
            raise ValueError('Counter assets cannot use symbolic links.')
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError('Counter asset must be a regular file.')
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DeepSeekV41Counter:
    def __init__(self, asset_manifest, provider_config, verification_path=None):
        self.manifest_path = Path(asset_manifest).absolute()
        self.manifest_hash = file_hash(self.manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text())
        self.config = json.loads(canonical(provider_config))
        if (self.manifest['schema'] != 'q01.counter.assets.v1'
                or self.manifest['model'] != MODEL or self.manifest['recipe_version'] != '0.2.0'
                or self.manifest['provider_config_sha256'] != digest(self.config)
                or set(self.config['models'].values()) != {MODEL}
                or any(p != {'thinking': {'type': 'disabled'}, 'temperature': 0}
                       for p in self.config['parameters'].values())):
            raise ValueError('Counter is restricted to the independently registered exact provider profile.')
        self.files = {k: ROOT / v['path'] for k, v in self.manifest['files'].items()}
        self._assert_assets()
        vendor = ROOT / self.manifest['vendor_path']
        if str(vendor) not in sys.path:
            sys.path.insert(0, str(vendor))
        recipe = importlib.import_module('deepseek_recipe')
        native = importlib.import_module('deepseek_recipe._native')
        if (Path(recipe.__file__).absolute() != self.files['python_init'].absolute()
                or Path(native.__file__).absolute() != self.files['native'].absolute()
                or recipe.__version__ != self.manifest['recipe_version']):
            raise ValueError('Actual loaded recipe library does not match its registered artifact.')
        self.recipe = recipe
        tokenizer = recipe.Tokenizer.from_file(str(self.files['tokenizer']))
        self.encoding = recipe.DeepseekV41Encoding().with_tokenizer(tokenizer)
        self.options = recipe.ConversionOptions(default_thinking_mode=False).with_model_alias(
            MODEL, recipe.ModelVersion.DeepseekV41Flash)
        literals = {'｜DSML｜', '<｜image｜>', '<｜deepseek_image｜>'}
        for name in ('tokenizer', 'recipe_tokenizer'):
            literals.update(t['content'] for t in json.loads(self.files[name].read_text())['added_tokens'])
        self.literals = tuple(sorted(literals))
        self.verification_path = Path(verification_path).absolute() if verification_path else None
        if self.verification_path is not None:
            certificate = json.loads(self.verification_path.read_text())
            if (certificate.get('approved') is not True
                    or certificate.get('asset_manifest_sha256') != self.manifest_hash
                    or certificate.get('implementation_sha256') != file_hash(__file__)
                    or certificate.get('provider_config_sha256') != digest(self.config)):
                raise ValueError('Independent verification certificate does not bind these exact artifacts.')

    def _assert_assets(self):
        if file_hash(self.manifest_path) != self.manifest_hash:
            raise ValueError('Counter asset manifest changed.')
        for name, entry in self.manifest['files'].items():
            if file_hash(self.files[name]) != entry['sha256']:
                raise ValueError('Counter artifact changed: ' + name)

    @property
    def identity(self):
        identity = {'kind': 'verified_chat_token_counter' if self.verification_path else 'unapproved_chat_token_counter',
            'scope': 'complete_chat_messages', 'verified_models': [MODEL],
            'provider_config_sha256': digest(self.config),
            'artifact_path': str(self.files['tokenizer'].absolute()),
            'artifact_sha256': self.manifest['files']['tokenizer']['sha256'],
            'chat_template_path': str(self.files['chat_template'].absolute()),
            'chat_template_sha256': self.manifest['files']['chat_template']['sha256'],
            'implementation_path': str(Path(__file__).absolute()), 'implementation_sha256': file_hash(__file__),
            'asset_manifest_path': str(self.manifest_path), 'asset_manifest_sha256': self.manifest_hash,
            'recipe_commit': self.manifest['recipe_commit'], 'recipe_version': self.recipe.__version__,
            'model_revision': self.manifest['model_revision'],
            'profile': copy.deepcopy(self.manifest['profile']), 'literal_rejection_sha256': digest(list(self.literals)),
            'actual_provider_count_equivalence_proven': False,
            'limit': 'Official local V4.1 reference on registered plain-text profile; literal special tokens rejected; live usage still separately recorded.'}
        if self.verification_path:
            identity.update(verification_path=str(self.verification_path),
                            verification_sha256=file_hash(self.verification_path))
        return identity

    def count_messages(self, messages, model):
        self._assert_assets()
        if (model != MODEL or type(messages) is not list or len(messages) != 2
                or any(type(m) is not dict or set(m) != {'role', 'content'} for m in messages)
                or [m['role'] for m in messages] != ['system', 'user']
                or any(type(m['content']) is not str for m in messages)):
            raise ValueError('Only the registered full system/user text profile is supported.')
        for message in messages:
            message['content'].encode('utf-8')
            if any(literal in message['content'] for literal in self.literals):
                raise ValueError('Ambiguous special-token literal in message; no rewrite or request permitted.')
        request = {'model': model, 'messages': copy.deepcopy(messages),
                   'thinking': {'type': 'disabled'}, 'response_format': {'type': 'json_object'}}
        converted = self.recipe.ChatCompletionRequest(request).convert(self.options)
        tokens = self.encoding.encode(converted.conversation)
        return {'input_tokens': len(tokens), 'model': model, 'messages_sha256': digest(messages),
                'chat_template_sha256': self.manifest['files']['chat_template']['sha256']}
