"""Portable entry point for the classroom demonstration and offline checks."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parent
DEMO = ROOT / 'runs/i01/20261106_200240_english_demo_rev053_287f5f2b'
sys.path.insert(0, str(DEMO))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=7862)
    parser.add_argument('--check', action='store_true', help='Check pinned inputs and replay without network calls.')
    parser.add_argument('--require-assets', action='store_true', help='Also require the downloaded classifier weight.')
    args = parser.parse_args()
    from live_demo import Engine, PROJECT, i01_knowledge, render
    engine = Engine()
    engine.check_binding(require_external=args.require_assets)
    if args.check:
        knowledge = i01_knowledge.validate_bundle(manifest_sha256=engine.policy['knowledge_manifest_sha256'])
        replay = {}
        for key in engine.cases:
            record = engine.replay(key)
            values = render(record)
            if len(values) != 7 or not all(isinstance(value, str) and value for value in values):
                raise ValueError('Incomplete replay: ' + key)
            replay[key] = {'predicted_class': values[1], 'task': values[2], 'origin': record.get('origin', 'historical')}
        binding = json.loads((DEMO / 'binding.json').read_bytes())
        print(json.dumps({'status': 'PASS', 'replays': replay, 'knowledge_counts': knowledge['counts'],
                          'source_profile': knowledge['source_distribution_profile'],
                          'missing_external_assets': [p for p in binding.get('external_files', {}) if not (PROJECT / p).is_file()],
                          'model_api_calls': 0, 'fresh_classifier_inference': False}, indent=2))
        return
    spec = importlib.util.spec_from_file_location('submission_demo_ui', DEMO / 'app.py')
    ui = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ui)
    import gradio as gr
    ui.build_page(engine).queue(max_size=1, default_concurrency_limit=1).launch(
        server_name='127.0.0.1', server_port=args.port, share=False, inbrowser=False,
        allowed_paths=[str(DEMO / 'assets')], strict_cors=True, show_error=False,
        footer_links=[], js=ui.english_locale_js(),
        theme=gr.themes.Default(font=['system-ui'], font_mono=['system-ui']))


if __name__ == '__main__':
    main()
