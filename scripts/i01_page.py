"""I01-S01: one registered local case, one durable batch, no arbitrary input."""
import argparse
import csv
from decimal import Decimal
from pathlib import Path
import threading
import time

import q01_flow as flow
import q01_intent as intent
import i01_http as wire
from q01_token_counter import DeepSeekV41Counter

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'runs/d03/20261022_rev006'
COMPATIBILITY = ROOT / 'runs/d03_fix/rev007/compatibility.json'
SCENARIO = 'SC-K01-OR-001'
_LOCK = threading.Lock()


def counter_for(profile):
    assets = ROOT / 'runs/q01_provider/20261025_tokenizer_b03d7e91'
    return DeepSeekV41Counter(assets / 'verification/counter_assets.json', profile,
                              assets / 'audit/counter_certificate.json')


def classify(run):
    """Fresh pixels -> original extractor -> original saved Pipeline.predict."""
    import joblib
    import numpy as np
    import d03_pipeline as d03
    registration = wire.read(run / 'inputs/sample_registration.json')
    row = registration['selected_row']
    for path, expected in registration['file_hashes'].items():
        if flow.file_hash(ROOT / path) != expected:
            raise ValueError('Registered classifier input changed.')
    if row['split'] not in ('train', 'validation'):
        raise ValueError('Only registered training/validation input allowed.')
    d03.check_frozen_run(MODEL, COMPATIBILITY)
    config = d03.config()
    device = d03.setup(config)
    started = time.perf_counter()
    extractor, identity = d03.load_extractor(MODEL / 'weights/resnet18-f37072fd.pth', config, device)
    frozen_weights = d03.read_json(MODEL / 'freeze.json')['weights']
    if any(identity[k] != frozen_weights[k] for k in identity):
        raise ValueError('Loaded extractor identity differs from original freeze.')
    features = d03.extract_rows(extractor, [row], config, device)
    pipeline = joblib.load(MODEL / 'model/pipeline.joblib')
    if not np.array_equal(pipeline.classes_, np.arange(4)):
        raise ValueError('Frozen Pipeline class order changed.')
    predictions, _, _ = d03.prediction_rows(pipeline, features, [row], config)
    return {'kind': 'fresh_image_inference', 'sample_id': row['sample_id'],
            'predicted_class': predictions[0]['predicted_class'], 'prediction': predictions[0],
            'feature_sha256': d03.array_hash(features), 'weight_identity': identity,
            'model_version': config['model_version'], 'elapsed_seconds': time.perf_counter() - started}


class SavedQA:
    def __init__(self, actual, run):
        self.actual, self.run, self.sequence = actual, run, 0

    @property
    def identity(self):
        return self.actual.identity

    def call(self, action, payload):
        self.sequence += 1
        try:
            return getattr(self.actual, action)(payload)
        finally:
            flow.write_record(self.run / 'qa_stages' / f'{self.sequence}_{action}.json', self.actual.snapshot())

    def generate(self, payload): return self.call('generate', payload)
    def verify(self, payload): return self.call('verify', payload)
    def revise(self, payload): return self.call('revise', payload)


def run_case(run):
    result = {'classifier': None, 'routing': None, 'flow_record': None,
              'status': 'input_or_setup_failed', 'service_message': None,
              'summary': {'identify_posts': 0, 'qa_posts': 0, 'total_posts': 0,
                          'reserved_cost_usd': '0', 'estimated_cost_usd': 'unknown', 'actual_bill': 'unknown'}}
    router = transport = slot = budget = qa = None
    try:
        # The release is checked before any inference, then frozen before HTTP.
        profile = wire.read(ROOT / 'configs/q01_deepseek.json')
        counter = counter_for(profile)
        wire.check_release(run, counter)
        result['classifier'] = prediction = classify(run)
        flow.write_record(run / 'classifier_record.json', prediction)
        if prediction['predicted_class'] != 'outer_race':
            result.update(status='knowledge_not_covered', service_message='当前知识未覆盖；本次预测保留，不更换样本。')
            return result
        approved = wire.read(run / 'inputs/approved_content.json')
        four = dict(approved, predicted_class=prediction['predicted_class'], scenario_id=SCENARIO)
        flow.write_record(run / 'inputs/four_fields.json', four)
        gate = wire.LiveGate(run, counter, prediction)
        gate.freeze()
        slot = wire.IntentSlot(run / 'intent_slot', flow.digest(counter.identity))
        transport = wire.IntentHTTP(profile, gate.intent_config, gate.prompts, slot, gate, token_counter=counter)
        router = intent.ModelTaskRouter(profile, gate.intent_config, gate.prompts, transport.request,
                                        transport_kind='real_http')
        route = router.prepare_request(gate.projection)
        result['routing'] = route
        provider_snapshot, http_snapshot = router.snapshot(), transport.snapshot()
        flow.write_record(run / 'recognition_record.json', route)
        flow.write_record(run / 'recognition_provider_snapshot.json', provider_snapshot)
        flow.write_record(run / 'recognition_http_snapshot.json', http_snapshot)
        if route['status'] != 'ready':
            result.update(status=route['status'], service_message=route['service_message'])
            return result
        flow.write_record(run / 'derived_request.json', route['request'])
        gate.bind_derived(route, provider_snapshot, http_snapshot)
        sha = flow.file_hash(run / 'derived_request.json')
        budget = wire.ChainQABudget(run / 'qa_session', gate.release['qa_budget'], sha)
        (run / 'qa_stages').mkdir(exist_ok=False)
        qa = wire.DerivedQAProvider(profile, budget, sha, gate=gate, token_counter=counter)
        runtime = {'record_kind': 'real_provider', 'prompts': wire.read(ROOT / 'configs/q01_provider_prompts.json'),
                   'generation_settings': {'provider': 'DeepSeek', 'provider_config_sha256': flow.digest(profile)}}
        record = flow.run_flow(route['request'], flow.group_config('GV'), SavedQA(qa, run), runtime=runtime)
        record['provider_observation'] = qa.snapshot()
        record['source_task_recognition_sha256'] = flow.file_hash(run / 'recognition_record.json')
        flow.write_record(run / 'qa_record.json', record)
        result.update(flow_record=record, status=record['status'], service_message=record['service_message'])
    except Exception as error:
        # Exception text can contain credentials; sending layers retain redacted diagnostics.
        result.update(status='input_or_setup_failed', service_message='本批次已停止；请查看保存的失败代码。')
        result['failure'] = {'error_type': type(error).__name__,
                             'error_code': getattr(error, 'code', 'setup_or_recording_failure')}
    finally:
        finalization_errors = []

        def save_final(filename, snapshot):
            try:
                flow.write_record(run / filename, snapshot())
            except Exception as error:
                finalization_errors.append({'stage': filename, 'error_type': type(error).__name__})

        summary = result['summary']
        if transport is not None:
            summary['identify_posts'] = transport.httpx_post_invocations
        if qa is not None:
            summary.update(qa_posts='unknown', total_posts='unknown')
        if slot is not None:
            summary['reserved_cost_usd'] = slot.reserved if budget is None else 'unknown'
        try:
            attempts = ([] if router is None else router.attempts) + ([] if qa is None else qa.attempts)
            estimate = Decimal(0)
            known = bool(attempts)
            for attempt in attempts:
                usage = attempt.get('token_usage')
                if type(usage) is dict and all(type(usage.get(k)) is int and usage[k] >= 0 for k in ('prompt_tokens','completion_tokens')):
                    estimate += (Decimal(usage['prompt_tokens']) * Decimal('.30') + Decimal(usage['completion_tokens']) * Decimal('1.20')) / 1000000
                else:
                    known = False
            qa_snapshot = None if budget is None else budget.snapshot()
            summary.update(qa_posts=0 if qa is None else qa.snapshot()['actual_attempt_count'],
                           estimated_cost_usd=str(estimate) if known else 'unknown',
                           reserved_cost_usd=str((Decimal(0) if slot is None else Decimal(slot.reserved)) +
                                                (Decimal(0) if qa_snapshot is None else Decimal(qa_snapshot['reserved_cost']))))
            summary['total_posts'] = summary['identify_posts'] + summary['qa_posts']
            if summary['total_posts'] > 5 or Decimal(summary['reserved_cost_usd']) > Decimal('.02'):
                raise RuntimeError('Hard ceiling invariant violated.')
            if qa_snapshot is not None:
                save_final('qa_budget_after.json', lambda: qa_snapshot)
        except Exception as error:
            finalization_errors.append({'stage': 'final_summary', 'error_type': type(error).__name__})
        finally:
            if budget is not None:
                try:
                    budget.close()
                except Exception as error:
                    finalization_errors.append({'stage': 'budget_close', 'error_type': type(error).__name__})
        if transport is not None:
            save_final('final_identify_http.json', transport.snapshot)
        if qa is not None:
            save_final('final_qa_provider.json', qa.snapshot)
        if finalization_errors:
            result.update(status='input_or_setup_failed', service_message='本批次收尾记录或清理失败；不会重新发送。')
            result['failure'] = {'error_type': finalization_errors[0]['error_type'],
                                 'error_code': 'final_recording_failed'}
            result['finalization_errors'] = finalization_errors
    return result


def execute_once(run):
    """Exclusive O_EXCL claim survives repeats, process restarts and failures."""
    run = wire.old._no_links(run)
    if not run.is_relative_to(ROOT / 'runs/i01'):
        raise ValueError('A registered I01 run is required.')
    if not _LOCK.acquire(blocking=False):
        return {'status': 'batch_consumed', 'service_message': '本批次正在执行，重复点击不会发送请求。'}
    try:
        saved = run / 'page_record.json'
        if saved.exists():
            return wire.read(saved)
        try:
            wire.put(run / 'live_batch_reserved.json', {'release_sha256': flow.file_hash(run / 'live_release.json'),
                     'scope': 'I01-S01 one registered image, one identify and one derived GV; no retry/refund/resume.'})
        except FileExistsError:
            return {'status': 'batch_consumed', 'service_message': '本批次已经使用或中断；不能重新发送。'}
        try:
            result = run_case(run)
        except Exception as error:
            result = {'status': 'input_or_setup_failed', 'classifier': None, 'routing': None,
                      'flow_record': None, 'service_message': '本批次异常停止；不会重新发送。',
                      'failure': {'error_type': type(error).__name__, 'error_code': 'case_failed'},
                      'summary': {k: 'unknown' for k in ('identify_posts', 'qa_posts', 'total_posts',
                                  'reserved_cost_usd', 'estimated_cost_usd', 'actual_bill')}}
        wire.put(saved, result)
        return result
    finally:
        _LOCK.release()


def render(result):
    """Product fields only; drafts and internal hashes never rendered."""
    classifier = result.get('classifier') or {}
    route = result.get('routing') or {}
    record = result.get('flow_record') or {}
    status = result.get('status', 'not_started')
    labels = {'evidence_insufficient': '证据不足', 'knowledge_not_covered': '当前知识未覆盖',
              'completed_verified_model_report': '已完成模型核验', 'verification_failed': '未完成核验',
              'content_verification_failed': '未完成核验', 'content_issues_remaining': '未完成核验',
              'schema_failed': '未完成核验', 'citation_failed': '未完成核验',
              'revision_failed': '未完成核验', 'generation_failed': '无法完成本次请求',
              'retrieval_failed': '无法完成本次请求', 'invalid_input': '无法完成本次请求',
              'needs_clarification': '任务识别阻断：需要澄清',
              'unsupported': '任务识别阻断：不支持', 'recognition_failed': '任务识别失败',
              'input_or_setup_failed': '无法完成本次请求', 'batch_consumed': '本批次已使用'}
    predicted = classifier.get('predicted_class', '尚未识别')
    scenario = '尚未映射'
    if predicted == 'outer_race':
        with (ROOT / 'knowledge/k01/scenarios.csv').open(newline='', encoding='utf-8-sig') as source:
            scenario = next(r['assumption_text'] for r in csv.DictReader(source) if r['scenario_id'] == SCENARIO)
    saved_scenario = (record.get('retrieval') or {}).get('information_separation', {}).get('demo_scenario') or {}
    scenario = saved_scenario.get('assumption_text', scenario)
    tasks = (route.get('request') or {}).get('task_types', [])
    answer = record.get('final_answer') if record.get('publishable') is True and not result.get('failure') else None
    cards = record.get('returned_cards') or []
    evidence_blocks = []
    for card in cards:
        source = card.get('source') or {}
        evidence_blocks.append('\n\n'.join([
            f"**{card.get('evidence_id', '')} · {source.get('title', '')}**",
            card.get('original_text', ''),
            f"出处：[{source.get('organization_author', '')}]({card.get('source_url', '')})；"
            f"{card.get('source_version', '')}；{card.get('page_or_section', '')}",
            f"条件：{card.get('conditions', '')}",
            f"限制：{card.get('limitations', '')}"]))
    evidence = '\n\n---\n\n'.join(evidence_blocks)
    state = labels.get(status, '未完成核验' if 'fail' in status else status)
    message = result.get('service_message') or record.get('reason') or ''
    return (predicted, scenario, ', '.join(tasks) or '尚未识别',
            f'**{state}**\n\n{message}\n\n本页仅验证当前单情景接线；模型核验不证明维护正确性。',
            answer or '本次没有可发布答案。', evidence or '尚未取得原文证据。')


def build_page(run):
    import gradio as gr
    registration = wire.read(run / 'inputs/sample_registration.json')
    approved = wire.read(run / 'inputs/approved_content.json')
    saved = wire.read(run / 'page_record.json') if (run / 'page_record.json').exists() else {}
    with gr.Blocks(title='AAE5103 单情景演示', analytics_enabled=False) as page:
        gr.Markdown('# AAE5103 当前外圈单情景演示\n\n固定验证样本与已登记问题。本批次只执行一次，重复点击只显示已保存结果。')
        gr.Image(value=str(ROOT / registration['selected_row']['model_image_path']), label='已登记的标准时频图', interactive=False)
        gr.Textbox(value=approved['question'], label='已登记开发问题', interactive=False)
        gr.JSON(value=approved['explicitly_allowed_operating_context'], label='获准工况', open=False)
        button = gr.Button('运行本次完整案例', variant='primary')
        predicted = gr.Textbox(label='本次分类器预测', interactive=False)
        scenario = gr.Textbox(label='演示情景假设', interactive=False)
        tasks = gr.Textbox(label='本次模型识别任务', interactive=False)
        status = gr.Markdown()
        answer = gr.Markdown(label='可发布答案')
        with gr.Accordion('原文证据与出处', open=True):
            evidence = gr.Markdown()
        outputs = [predicted, scenario, tasks, status, answer, evidence]
        for component, value in zip(outputs, render(saved)):
            component.value = value
        def click():
            return render(execute_once(run))
        button.click(click, inputs=[], outputs=outputs, api_name='run_case', concurrency_limit=1)
    return page


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    run = wire.old._no_links(Path(args.run).absolute())
    page = build_page(run)
    page.launch(server_name='127.0.0.1', server_port=7860, share=False,
                allowed_paths=[str(ROOT / wire.read(run / 'inputs/sample_registration.json')['selected_row']['model_image_path'])],
                strict_cors=True, show_error=False)


if __name__ == '__main__':
    main()
