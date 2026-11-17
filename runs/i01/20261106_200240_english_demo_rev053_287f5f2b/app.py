"""English-only live demonstration and saved-run replay; no calls on startup."""
from pathlib import Path
import argparse
import os

ROOT = Path(__file__).absolute().parent
os.environ['NO_PROXY'] = '127.0.0.1,localhost'
os.environ['no_proxy'] = '127.0.0.1,localhost'
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ['GRADIO_TEMP_DIR'] = str(ROOT / 'runtime/gradio')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from live_demo import DemoError, Engine, render
from english_display import english


def build_page(engine):
    import gradio as gr
    key = 'demo_2'
    case = engine.cases[key]
    with gr.Blocks(title='AAE5103 Bearing Classification and Evidence-Based Q&A', analytics_enabled=False) as page:
        gr.Markdown('# Bearing Classification and Evidence-Based Q&A\nRun a new workflow or replay a saved result.')
        with gr.Row():
            with gr.Column(scale=1):
                choice = gr.Dropdown([(c['title'], k) for k, c in engine.cases.items()],
                                     value=key, label='Demonstration Case', filterable=False)
                image = gr.Image(value=str(ROOT / case['image']), label='Validation Time-Frequency Image',
                                 interactive=False, buttons=[])
                question = gr.Textbox(value=case['question'], label='English Question', interactive=False, lines=3)
                live = gr.Button('Run Live Demo', variant='primary')
                replay = gr.Button('Replay Saved Record')
                gr.Markdown('Local SVM classification -> API task identification -> knowledge graph retrieval -> '
                            'API evidence-based answer. Maximum two API requests per live run.')
            with gr.Column(scale=2):
                progress = gr.Markdown('Run the live demo to see the status and duration of each stage.')
                predicted = gr.Textbox(label='SVM Predicted Class', interactive=False)
                tasks = gr.Textbox(label='Question Task Identified by the API', interactive=False)
                with gr.Accordion('Independent Teaching Scenario', open=False):
                    scenario = gr.Markdown('No scenario has been selected yet.')
                ready = bool(os.environ.get('DEEPSEEK_API_KEY', '').strip())
                status = gr.Markdown('Ready. The live button sends new model requests; replay does not.' if ready else
                                     '**No API key is configured for this service.** Start the launcher from a terminal '
                                     'with DEEPSEEK_API_KEY configured. Saved replay is available without a key.')
                answer = gr.Markdown('No answer has been generated yet.')
        with gr.Accordion('Retrieved Evidence and Original Sources', open=True):
            evidence = gr.Markdown('No evidence has been retrieved yet.')
        gr.Markdown('Educational prototype. An image class does not establish a damage cause. Answers are limited '
                    'to the returned evidence and do not authorize maintenance or equipment release.')
        outputs = [progress, predicted, tasks, scenario, evidence, answer, status]

        def run(key):
            try:
                for record in engine.live(key):
                    yield (*render(record), gr.update(interactive=False), gr.update(interactive=False),
                           gr.update(interactive=False))
            except Exception as error:
                message = english(str(error), 'Demonstration preparation failed. No automatic retry was made.') if isinstance(error, DemoError) else 'Demonstration preparation failed. No automatic retry was made.'
                yield ('This run did not start.', 'Not classified', 'Not identified', 'No scenario selected.',
                       'No evidence has been retrieved.', 'No answer is available for display.', message,
                       gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True))
                return
            yield (*render(record), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True))

        def saved(key):
            try:
                return render(engine.replay(key))
            except Exception as error:
                message = english(str(error), 'The saved record could not be read.') if isinstance(error, DemoError) else 'The saved record could not be read.'
                return ('Replay failed.', 'Not classified', 'Not identified', 'No scenario selected.',
                        'No evidence is available for display.', 'No answer is available for display.', message)

        def select(key):
            case = engine.cases[key]
            return (str(ROOT / case['image']), case['question'], 'Ready to start.', 'Not classified',
                    'Not identified', 'No scenario selected.', 'No evidence has been retrieved yet.',
                    'No answer has been generated yet.', 'Choose Run Live Demo or Replay Saved Record.')

        live.click(run, choice, [*outputs, live, replay, choice], api_name='live_demo',
                   concurrency_limit=None, trigger_mode='once')
        replay.click(saved, choice, outputs, api_name='saved_replay', concurrency_limit=1, concurrency_id='demo')
        choice.change(select, choice, [image, question, *outputs], api_name='select_case')
    return page


def english_locale_js():
    """Use the installed Gradio locale API so built-in controls are English too."""
    import gradio as gr
    assets = Path(gr.__file__).parent / 'templates/frontend/assets'
    for asset in sorted(assets.glob('core-*.js')):
        if 'as changeLocale' in asset.read_text():
            return ('async () => { const { changeLocale } = await import("/assets/' + asset.name
                    + '"); changeLocale("en"); document.documentElement.lang = "en"; }')
    raise RuntimeError('English locale support is unavailable in the installed Gradio frontend.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7862)
    args = parser.parse_args()
    engine = Engine()
    engine.check_binding()
    build_page(engine).queue(max_size=1, default_concurrency_limit=1).launch(
        server_name='127.0.0.1', server_port=args.port, share=False, inbrowser=False,
        allowed_paths=[str(ROOT / 'assets')], strict_cors=True, show_error=False,
        footer_links=[], js=english_locale_js(),
        theme=__import__('gradio').themes.Default(font=['system-ui'], font_mono=['monospace']))
