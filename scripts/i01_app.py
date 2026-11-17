"""Local, single-turn I01 page over the frozen classifier and accepted QA core."""
import argparse
from decimal import Decimal
from pathlib import Path
import threading

import i01_page as single
import i01_session as wire
import i01_knowledge as knowledge
import q01_flow as flow
import q01_intent as intent

ROOT = single.ROOT


class PredictionContext:
    """Project bound live origins without changing static knowledge or cards."""
    def __init__(self, retriever, classifier, route):
        classifier, route = flow.json_snapshot(classifier), flow.json_snapshot(route)
        request = route.get('request')
        identity = route.get('identity') or {}
        report = intent.validate_report(route.get('model_report'))
        if (classifier.get('kind') != 'fresh_image_inference' or route.get('status') != 'ready'
                or identity.get('kind') != 'real_provider' or identity.get('transport') != 'real_http'
                or identity.get('role') != 'task_identification' or report['status'] != 'identified'
                or type(request) is not dict):
            raise ValueError('Actual image inference and current successful task recognition are required.')
        flow.k02.validate_request(request)
        predicted = classifier.get('predicted_class')
        if (predicted not in knowledge.SCENARIOS or request['predicted_class'] != predicted
                or request['scenario_id'] != knowledge.SCENARIOS[predicted]
                or request['task_types'] != [t for t in flow.k02.TASKS if t in report['task_types']]):
            raise ValueError('Prediction, scenario and model-derived tasks must match this request.')
        self.retriever, self.request = retriever, request

    def retrieve(self, request, mode='text', top_k=3):
        if request != self.request:
            raise ValueError('The bound business request changed.')
        returned = flow.json_snapshot(self.retriever.retrieve(request, mode=mode, top_k=top_k))
        if returned.get('request') != self.request:
            raise ValueError('Retrieval is not bound to this business request.')
        separation = returned.get('information_separation') or {}
        scenario = separation.get('demo_scenario')
        if scenario is None and not returned.get('results'):
            return returned
        if (type(scenario) is not dict or scenario.get('predicted_class') != request['predicted_class']
                or scenario.get('scenario_id') != request['scenario_id']):
            raise ValueError('Retrieved scenario differs from the actual prediction route.')
        separation['static_class_semantics'] = scenario.get('class_semantics')
        separation['class_origin'] = 'frozen_classifier_fresh_image_inference'
        separation['task_origin'] = 'current_model_task_identification'
        scenario['class_semantics'] = (
            'The request predicted_class is the actual output of this invocation of the frozen '
            'image classifier, not a manual stub. It supports only class/location routing and '
            'does not establish a damage mechanism, root cause, absence of damage or equipment safety. '
            'The request task_types come from the current model task-identification result. '
            'Static scenario-table class semantics describe a routing definition; the scenario '
            'cause remains an independent educational assumption, not an observed diagnosis. '
            'Static statements about unbound samples or class-only routing describe the reusable '
            'scenario definition and do not deny this classifier invocation. Operating facts '
            'remain limited to explicitly_allowed_operating_context.')
        returned['information_separation'] = separation
        return returned


def failure(code, message, error=None):
    return {'status': 'input_or_setup_failed', 'service_message': message,
            'failure': {'error_code': code, 'error_type': type(error).__name__ if error else 'ControlledStop'},
            'summary': {k: 'unknown' for k in ('identify_posts', 'qa_posts', 'total_posts',
                        'reserved_cost_usd', 'estimated_cost_usd', 'actual_bill')}}


def run_case(case, submission, counter, retriever, batch):
    result = {'classifier': None, 'routing': None, 'flow_record': None,
              'status': 'input_or_setup_failed', 'service_message': None,
              'summary': {'identify_posts': 0, 'qa_posts': 0, 'total_posts': 0,
                          'reserved_cost_usd': '0', 'estimated_cost_usd': 'unknown', 'actual_bill': 'unknown'}}
    router = transport = slot = budget = qa = None
    try:
        checked = wire.check_release(batch, counter)
        profile = checked['profile']
        result['classifier'] = prediction = single.classify(case)
        flow.write_record(case / 'classifier_record.json', prediction)
        predicted = prediction['predicted_class']
        if predicted not in knowledge.SCENARIOS:
            result.update(status='knowledge_not_covered', service_message='当前知识未覆盖；保留本次预测，不更换样本。')
            return result
        scenario_id = knowledge.SCENARIOS[predicted]
        result['scenario'] = retriever.scenarios[scenario_id]
        four = {'predicted_class': predicted, 'scenario_id': scenario_id,
                'question': submission['question'],
                'explicitly_allowed_operating_context': submission['explicitly_allowed_operating_context']}
        flow.write_record(case / 'inputs/four_fields.json', four)
        gate = wire.LiveGate(case, counter, prediction, batch_run=batch, actual_submission=submission)
        gate.freeze()
        slot = wire.IntentSlot(case / 'intent_slot', flow.digest(counter.identity))
        transport = wire.IntentHTTP(profile, gate.intent_config, gate.prompts, slot, gate, token_counter=counter)
        router = intent.ModelTaskRouter(profile, gate.intent_config, gate.prompts, transport.request,
                                       transport_kind='real_http')
        route = router.prepare_request(gate.projection)
        result['routing'] = route
        provider_snapshot, http_snapshot = router.snapshot(), transport.snapshot()
        for name, value in (('recognition_record.json', route),
                            ('recognition_provider_snapshot.json', provider_snapshot),
                            ('recognition_http_snapshot.json', http_snapshot)):
            flow.write_record(case / name, value)
        if route['status'] != 'ready':
            result.update(status=route['status'], service_message=route['service_message'])
            return result
        flow.write_record(case / 'derived_request.json', route['request'])
        gate.bind_derived(route, provider_snapshot, http_snapshot)
        prediction_context = PredictionContext(retriever, prediction, route)
        sha = flow.file_hash(case / 'derived_request.json')
        budget = wire.ChainQABudget(case / 'qa_session', gate.release['qa_budget'], sha)
        (case / 'qa_stages').mkdir(exist_ok=False)
        qa = wire.DerivedQAProvider(profile, budget, sha, gate=gate, token_counter=counter)
        runtime = {'record_kind': 'real_provider', 'prompts': wire.read(ROOT / 'configs/i01_provider_prompts.json'),
                   'generation_settings': {'provider': 'DeepSeek', 'provider_config_sha256': flow.digest(profile)}}
        record = flow.run_flow(route['request'], flow.group_config('GV'), single.SavedQA(qa, case),
                               retriever=prediction_context, runtime=runtime)
        # Only identity diagnostics change: business fields and full cards are the core's actual return.
        record['identity']['expected_k02_knowledge_hashes'] = retriever.knowledge_hashes
        record['identity']['knowledge_binding_note'] = 'Externally pinned I01 extension; unchanged K02 algorithms.'
        record['identity']['i01_retriever'] = retriever.identity
        record['provider_observation'] = qa.snapshot()
        record['source_task_recognition_sha256'] = flow.file_hash(case / 'recognition_record.json')
        flow.write_record(case / 'qa_record.json', record)
        result.update(flow_record=record, status=record['status'], service_message=record['service_message'])
    except Exception as error:
        result.update(status='input_or_setup_failed', service_message='本次请求已停止；失败记录已保存，不会重新发送。')
        result['failure'] = {'error_type': type(error).__name__,
                             'error_code': getattr(error, 'code', 'setup_or_recording_failure')}
    finally:
        errors = []

        def save(name, getter):
            try:
                flow.write_record(case / name, getter())
            except Exception as error:
                errors.append({'stage': name, 'error_type': type(error).__name__})

        summary = result['summary']
        if transport is not None:
            summary['identify_posts'] = transport.httpx_post_invocations
        if qa is not None:
            summary.update(qa_posts='unknown', total_posts='unknown')
        if slot is not None:
            summary['reserved_cost_usd'] = slot.reserved if budget is None else 'unknown'
        try:
            attempts = ([] if router is None else router.attempts) + ([] if qa is None else qa.attempts)
            estimate, known = Decimal(0), bool(attempts)
            for attempt in attempts:
                usage = attempt.get('token_usage')
                if type(usage) is dict and all(type(usage.get(k)) is int and usage[k] >= 0
                                             for k in ('prompt_tokens', 'completion_tokens')):
                    estimate += (Decimal(usage['prompt_tokens']) * Decimal('.30') +
                                 Decimal(usage['completion_tokens']) * Decimal('1.20')) / 1000000
                else:
                    known = False
            snapshot = None if budget is None else budget.snapshot()
            summary.update(qa_posts=0 if qa is None else qa.snapshot()['actual_attempt_count'],
                           estimated_cost_usd=str(estimate) if known else 'unknown',
                           reserved_cost_usd=str((Decimal(0) if slot is None else Decimal(slot.reserved)) +
                                                (Decimal(0) if snapshot is None else Decimal(snapshot['reserved_cost']))))
            summary['total_posts'] = summary['identify_posts'] + summary['qa_posts']
            if summary['total_posts'] > 5 or Decimal(summary['reserved_cost_usd']) > Decimal('.02'):
                raise RuntimeError('Flow ceiling violated.')
            if snapshot is not None:
                save('qa_budget_after.json', lambda: snapshot)
        except Exception as error:
            errors.append({'stage': 'final_summary', 'error_type': type(error).__name__})
        finally:
            if budget is not None:
                try:
                    budget.close()
                except Exception as error:
                    errors.append({'stage': 'budget_close', 'error_type': type(error).__name__})
        if transport is not None:
            save('final_identify_http.json', transport.snapshot)
        if qa is not None:
            save('final_qa_provider.json', qa.snapshot)
        if errors:
            result.update(status='input_or_setup_failed', service_message='收尾记录或清理失败；不会重新发送。',
                          failure={'error_code': 'final_recording_failed', 'error_type': errors[0]['error_type']},
                          finalization_errors=errors)
    return result


def render(result):
    values = list(single.render(result))
    if result.get('scenario'):
        values[1] = result['scenario']['assumption_text'] + '\n\n适用限制：' + result['scenario']['applicability']
    values[3] = values[3].replace('本页仅验证当前单情景接线；', '本页用于四类、单轮知识查询原型；')
    if values[0] == 'normal':
        values[1] += '\n\nnormal 仅是分类器输出，不证明设备无损伤、适航或可放行；此分支只给一般监测知识。'
    return tuple(values)


class Workflow:
    def __init__(self, batch):
        self.batch = wire.old._no_links(batch)
        self.counter = single.counter_for(wire.read(ROOT / 'configs/q01_deepseek.json'))
        self.checked = wire.check_release(self.batch, self.counter)
        self.retriever = knowledge.Retriever(manifest_sha256=self.checked['release']['knowledge_manifest_sha256'])
        self.budget = wire.BatchBudget(self.batch)
        self.lock = threading.Lock()

    def submit(self, sample_choice, question):
        if not self.lock.acquire(blocking=False):
            return {'status': 'batch_consumed', 'service_message': '当前请求正在执行；重复操作不会发送。'}
        case = None
        try:
            wire.check_release(self.batch, self.counter)
            if sample_choice not in self.checked['registry']['samples']:
                return failure('invalid_input', '请选择已登记的样本。')
            intent.validate_input({'predicted_class': 'normal', 'scenario_id': knowledge.SCENARIOS['normal'],
                                   'question': question, 'explicitly_allowed_operating_context': {}})
            counted = self.counter.count_messages([
                {'role': 'system', 'content': self.checked['prompts']['instructions']},
                {'role': 'user', 'content': flow.canonical({'question': question})}], 'deepseek-flash')
            if counted['input_tokens'] > self.checked['intent_config']['max_input_tokens']:
                return failure('input_token_budget_exceeded', '问题超过识别输入预算；没有发送请求。')
            submission = {'sample_choice': sample_choice, 'question': question,
                          'explicitly_allowed_operating_context': {}}
            # Exact submissions are replayed from disk, including after a process restart.
            for previous in sorted((self.batch / 'cases').iterdir()):
                path = previous / 'inputs/submission.json'
                if path.exists() and wire.read(path) == submission:
                    saved = previous / 'page_record.json'
                    return wire.read(saved) if saved.exists() else {
                        'status': 'batch_consumed', 'service_message': '这个输入已使用或中断；不会重新发送。'}
            case = self.budget.claim_flow()
            raw = wire.put(case / 'inputs/submission.json', submission)
            wire.put(case / 'inputs/source_registration.json', {
                'path': (case / 'inputs/submission.json').relative_to(ROOT).as_posix(),
                'purpose': 'Exact current page single-turn input; registered before parsing; no E01 evaluation input.',
                'actual_sha256': wire.hashlib.sha256(raw).hexdigest()})
            wire.put(case / 'inputs/sample_registration.json', self.checked['registry']['samples'][sample_choice])
            try:
                result = run_case(case, submission, self.counter, self.retriever, self.batch)
            except Exception as error:
                result = failure('case_failed', '本次异常停止；不会重新发送。', error)
            result['case_id'] = case.name
            summary = result.get('summary', {})
            try:
                self.budget.finish_flow(case, result['status'],
                    identify_attempts=summary.get('identify_posts', 'unknown'),
                    qa_attempts=summary.get('qa_posts', 'unknown'))
            except Exception as error:
                result.update(status='input_or_setup_failed', service_message='批次收尾失败；不会重新发送。',
                              failure={'error_code': 'batch_finalization_failed', 'error_type': type(error).__name__})
            wire.put(case / 'page_record.json', result)
            # The committed page record is canonical. A redundant render copy
            # cannot change the result returned by this or a repeated click.
            try:
                wire.put(case / 'display_record.json', list(render(result)))
            except Exception as error:
                try:
                    wire.put(case / 'display_record_failure.json',
                             {'error_type': type(error).__name__, 'canonical_result_unchanged': True})
                except Exception:
                    pass
            return result
        except Exception as error:
            result = failure(getattr(error, 'code', 'submission_failed'), '输入或批次控制未通过；没有重试。', error)
            if case is not None and not (case / 'page_record.json').exists():
                wire.put(case / 'page_record.json', result)
            return result
        finally:
            self.lock.release()

    def budget_text(self):
        try:
            value = self.budget.snapshot()
        except Exception:
            return '批次控制不可用；不会继续发送。总上限 USD 0.24，异常额度不退还。'
        return f"本批次剩余 {value['remaining_flows']}/12 次。每流程上限 USD 0.02，总上限 USD 0.24；异常不退还额度。"


def build_page(workflow):
    import gradio as gr
    samples = workflow.checked['registry']['samples']
    examples = wire.read(workflow.batch / 'inputs/acceptance_requests.json')['cases']
    with gr.Blocks(title='AAE5103 本地知识查询', analytics_enabled=False) as page:
        gr.Markdown('# AAE5103 轴承状态与知识查询\n选择登记样本并输入完整英文问题。预测部位与预设原因分别展示；回答只依据页面列出的原文。')
        with gr.Row():
            choice = gr.Dropdown([(f'登记样本 {i}', key) for i, key in enumerate(samples, 1)],
                                 value=next(iter(samples)), label='标准时频图')
            image = gr.Image(value=str(ROOT / samples[next(iter(samples))]['selected_row']['model_image_path']),
                             interactive=False, label='模型输入图', buttons=[])
        example = gr.Dropdown([(f'开发示例 {i+1}', str(i)) for i in range(len(examples))], label='可选开发示例')
        question = gr.Textbox(label='英文问题（单轮）', value=examples[0]['question'], lines=3, max_length=2000)
        gr.Markdown('工况保持未知；不把样本标签、路径、文件名或实验负载送入问答。仅支持 explanation、inspection、preparation。')
        button = gr.Button('识别并查询', variant='primary')
        allowance = gr.Markdown(workflow.budget_text())
        prediction = gr.Textbox(label='本次预测类别', interactive=False)
        scenario = gr.Markdown('尚未映射')
        tasks = gr.Textbox(label='本次模型识别任务', interactive=False)
        status = gr.Markdown('尚未开始')
        answer = gr.Markdown('尚无可发布答案')
        evidence = gr.Markdown('尚未取得原文证据')

        def callback(c, q):
            result = workflow.submit(c, q)
            return (*render(result), workflow.budget_text())

        def select_example(index):
            row = examples[int(index)]
            return row['sample_choice'], row['question']

        choice.change(lambda c: str(ROOT / samples[c]['selected_row']['model_image_path']), choice, image,
                      api_name=False)
        example.change(select_example, example, [choice, question], api_name=False)
        button.click(callback, [choice, question], [prediction, scenario, tasks, status, answer, evidence, allowance],
                     api_name='query', concurrency_limit=1, concurrency_id='single_batch')
    return page


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    workflow = Workflow(args.run)
    page = build_page(workflow)
    allowed = [str(ROOT / row['selected_row']['model_image_path'])
               for row in workflow.checked['registry']['samples'].values()]
    try:
        page.launch(server_name='127.0.0.1', server_port=7861, share=False, inbrowser=False,
                    allowed_paths=allowed, strict_cors=True, show_error=False)
    finally:
        workflow.budget.close()


if __name__ == '__main__':
    main()
