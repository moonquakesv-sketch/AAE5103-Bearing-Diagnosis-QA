"""Distribution checks for offline operation and asset integrity; no provider calls."""
from pathlib import Path
import hashlib
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / 'runs/i01/20261106_200240_english_demo_rev053_287f5f2b'
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(DEMO))
import fetch_submission_assets as assets
import i01_knowledge as knowledge
import live_demo


class SubmissionPortability(unittest.TestCase):
    def test_both_saved_cases_render_without_provider_or_classifier(self):
        with (patch.object(live_demo.API, 'call', side_effect=AssertionError('Unexpected API call')),
              patch.object(live_demo.i01_page, 'classify', side_effect=AssertionError('Unexpected inference'))):
            engine = live_demo.Engine()
            engine.check_binding()
            for key in engine.cases:
                values = live_demo.render(engine.replay(key))
                self.assertIn('0 new API requests', values[6])
                self.assertTrue(all(values))

    def test_missing_weight_allows_replay_check_and_stops_live_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory).resolve()
            root = project / 'demo'
            root.mkdir()
            core = project / 'core.txt'
            core.write_text('SYNTHETIC immutable input')
            (root / 'binding.json').write_text(json.dumps({
                'files': {'core.txt': hashlib.sha256(core.read_bytes()).hexdigest()},
                'external_files': {'weights/model.pth': '0' * 64}}))
            engine = live_demo.Engine.__new__(live_demo.Engine)
            engine.root, engine.synthetic = root, False
            with patch.object(live_demo, 'PROJECT', project):
                engine.check_binding()
                with self.assertRaises(live_demo.DemoError) as missing:
                    engine.check_binding(require_external=True)
                self.assertEqual(missing.exception.code, 'asset_missing')
                core.write_text('SYNTHETIC changed input')
                with self.assertRaises(live_demo.DemoError) as changed:
                    engine.check_binding()
                self.assertEqual(changed.exception.code, 'input_changed')

    def test_exported_source_profile_retains_provenance_and_rejects_mismatch(self):
        pin = json.loads((DEMO / 'inputs/policy.json').read_bytes())['knowledge_manifest_sha256']
        retriever = knowledge.Retriever(manifest_sha256=pin)
        self.assertEqual(retriever.validation['counts']['evidence'], 16)
        self.assertEqual(retriever.validation['source_distribution_profile'], 'excerpt-only-v1')
        self.assertEqual(retriever.validation['local_excerpt_presence_checks'], [])
        self.assertEqual(len(retriever.validation['exported_excerpt_identity_checks']), 11)
        files = retriever.validation['file_hashes']
        blobs = {name: (ROOT / name).read_bytes() for name in files}
        index = json.loads((ROOT / 'knowledge/i01/v1/source_excerpt_index.json').read_bytes())
        first = next(iter(index['sources']))
        index['sources'][first]['snapshot_sha256'] = '0' * 64
        with self.assertRaises(knowledge.Error):
            knowledge._validate(retriever.tables, retriever.interface, blobs, files, excerpt_index=index)

    def test_downloader_rejects_changed_bytes_and_keeps_existing_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory).resolve()
            asset = {'path': 'weights/model.pth', 'url': 'https://example.invalid/fixture',
                     'sha256': hashlib.sha256(b'expected fixture').hexdigest()}
            with (patch.object(assets, 'ROOT', project),
                  patch.object(assets.urllib.request, 'urlopen', return_value=io.BytesIO(b'wrong fixture'))):
                with self.assertRaises(ValueError):
                    assets.fetch(asset)
                self.assertFalse((project / asset['path']).exists())
                self.assertEqual(list((project / 'weights').iterdir()), [])
                target = project / asset['path']
                target.write_bytes(b'existing different fixture')
                before = target.read_bytes()
                with self.assertRaises(ValueError):
                    assets.fetch(asset)
                self.assertEqual(target.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
