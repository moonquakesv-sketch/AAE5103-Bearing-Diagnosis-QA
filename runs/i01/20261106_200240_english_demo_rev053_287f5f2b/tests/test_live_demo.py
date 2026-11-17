"""SYNTHETIC control fixtures only; never sends model requests."""
from pathlib import Path
import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
import live_demo as demo

CARD = {"evidence_id": "EV-SYNTHETIC-001", "source": {"title": "SYNTHETIC", "organization_author": "SYNTHETIC"},
        "source_url": "https://example.invalid/SYNTHETIC", "page_or_section": "SYNTHETIC",
        "original_text": "SYNTHETIC visible preparation guidance.", "conditions": "SYNTHETIC", "limitations": "SYNTHETIC"}


class FakeRetriever:
    scenarios = {"SC-I01-RE-001": {"assumption_text": "SYNTHETIC", "assumed_cause": "SYNTHETIC",
                                    "applicability": "SYNTHETIC", "unknown_information": "SYNTHETIC"}}
    def retrieve(self, request, **kwargs):
        assert kwargs == {"mode": "graph", "top_k": 3}
        return {"results": [copy.deepcopy(CARD)]}


class FakeAPI:
    def __init__(self, bad_intent=False, bad_citation=False, fail_answer=False, chinese_answer=False):
        self.calls = []
        self.bad_intent, self.bad_citation, self.fail_answer = bad_intent, bad_citation, fail_answer
        self.chinese_answer = chinese_answer
    def call(self, folder, action, messages, calls):
        self.calls.append((action, messages))
        calls.append({"action": action, "usage": None, "SYNTHETIC": True})
        if action == "identify":
            assert json.loads(messages[1]["content"]) == {"question": "SYNTHETIC bearing preparation?"}
            if self.bad_intent:
                return "SYNTHETIC invalid JSON"
            return json.dumps({"status": "identified", "task_types": ["preparation"], "message": None})
        if self.fail_answer:
            raise demo.DemoError("timeout", "SYNTHETIC timeout")
        payload = json.loads(messages[1]["content"])
        assert payload["classifier_context"]["origin"] == "fresh_image_inference"
        assert payload["returned_cards"] == [CARD]
        assert "true_label" not in messages[1]["content"]
        if self.bad_citation:
            return "SYNTHETIC unsupported [EV-SYNTHETIC-999]."
        if self.chinese_answer:
            return '\u4e2d\u6587\u56de\u7b54 [EV-SYNTHETIC-001].'
        return 'SYNTHETIC tasks ["preparation"] require preparation [EV-SYNTHETIC-001].'


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="SYNTHETIC_", dir=ROOT / "tests")
        self.root = Path(self.temp.name)
        for folder in ["inputs", "runtime", "executions"]:
            (self.root / folder).mkdir()
        demo.write(self.root / "inputs/policy.json", {"cases": {"case": {"question": "SYNTHETIC bearing preparation?",
                   "registration": "inputs/sample.json"}}, "knowledge_manifest_sha256": "0" * 64})
        demo.write(self.root / "inputs/sample.json", {"SYNTHETIC": True})
        self.predictions = 0
    def tearDown(self):
        self.temp.cleanup()
    def classify(self, folder):
        self.predictions += 1
        return {"kind": "fresh_image_inference", "predicted_class": "rolling_element", "SYNTHETIC": True}
    def engine(self, api=None):
        return demo.Engine(self.root, classifier=self.classify, retriever_factory=FakeRetriever,
                           api=api or FakeAPI(), synthetic=True)
    def test_each_click_is_fresh_and_stages_are_visible(self):
        engine = self.engine()
        first = list(engine.live("case"))
        second = list(engine.live("case"))
        self.assertNotEqual(first[-1]["run_id"], second[-1]["run_id"])
        self.assertEqual(self.predictions, 2)
        self.assertEqual(len(engine.api.calls), 4)
        self.assertTrue(first[-1]["publishable"])
        self.assertEqual(first[-1]["origin"], "SYNTHETIC")
        self.assertEqual([s["status"] for s in first[-1]["stages"]], ["completed"] * 4)
        self.assertTrue(all(any(s["status"] == "running" for s in r["stages"]) for r in first[::2][:-1]))
        self.assertEqual(len(list((self.root / "executions").glob("*/record.json"))), 2)
    def test_failed_intent_prevents_retrieval_and_answer(self):
        api = FakeAPI(bad_intent=True)
        final = list(self.engine(api).live("case"))[-1]
        self.assertEqual(len(api.calls), 1)
        self.assertEqual(final["status"], "failed")
        self.assertIsNone(final["retrieval"])
        self.assertEqual(final["stages"][2]["status"], "skipped")
    def test_citations_accept_normal_brackets_but_reject_unknown_ids(self):
        good = list(self.engine().live("case"))[-1]
        self.assertIn('["preparation"]', good["answer"])
        bad = list(self.engine(FakeAPI(bad_citation=True)).live("case"))[-1]
        self.assertEqual(bad["failure"]["code"], "citation_failed")
        self.assertIsNone(bad["answer"])
        self.assertFalse(bad["publishable"])
    def test_failure_preserves_completed_stages(self):
        engine = self.engine(FakeAPI(fail_answer=True))
        final = list(engine.live("case"))[-1]
        self.assertEqual(final["failure"]["code"], "timeout")
        self.assertEqual(len(engine.api.calls), 2)
        self.assertEqual([s["status"] for s in final["stages"]], ["completed"] * 3 + ["failed"])
        self.assertIn("EV-SYNTHETIC-001", demo.render(final)[4])
        self.assertNotIn("unsupported", demo.render(final)[5])
    def test_non_English_response_is_retained_but_not_published_or_translated(self):
        engine = self.engine(FakeAPI(chinese_answer=True))
        final = list(engine.live('case'))[-1]
        self.assertEqual(final['failure']['code'], 'language_failed')
        self.assertEqual(len(engine.api.calls), 2)
        self.assertFalse(final['publishable'])
        raw = demo.read(self.root / 'executions' / final['run_id'] / 'generated_answer.json')['text']
        self.assertIsNotNone(demo.presentation.CHINESE.search(raw))
        self.assertTrue(all(demo.presentation.CHINESE.search(value) is None for value in demo.render(final)))
    def test_concurrent_click_rejected_and_cancel_releases_lock(self):
        engine = self.engine()
        generator = engine.live("case")
        next(generator)
        with self.assertRaises(demo.DemoError):
            list(engine.live("case"))
        generator.close()
        self.assertFalse(engine.lock.locked())
        self.assertEqual(list(engine.live("case"))[-1]["status"], "completed")
    def test_replay_is_zero_call_and_does_not_change_record(self):
        engine = self.engine()
        final = list(engine.live("case"))[-1]
        path = self.root / "executions" / final["run_id"] / "record.json"
        record = demo.read(path)
        record["origin"] = "LIVE"  # SYNTHETIC fixture emulates persisted live namespace only.
        path.write_text(json.dumps(record))
        digest = path.parent / "record_digest.json"
        digest.write_text(json.dumps({"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))
        before = path.read_bytes()
        count = len(engine.api.calls)
        saved = engine.replay("case")
        self.assertTrue(saved["replay"])
        self.assertEqual(count, len(engine.api.calls))
        self.assertEqual(before, path.read_bytes())
        self.assertIn("0 new API requests", demo.render(saved)[6])
        path.write_text("{}")
        with self.assertRaises(demo.DemoError):
            engine.replay("case")


class HTTPTests(unittest.TestCase):
    def test_response_retained_before_parse_failure_and_no_retry(self):
        import httpx
        with tempfile.TemporaryDirectory(prefix="SYNTHETIC_HTTP_", dir=ROOT / "tests") as tmp:
            folder = Path(tmp)
            transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b"SYNTHETIC not JSON",
                                                                         headers={"x-request-id": "SYNTHETIC-id"}))
            api = demo.API({"model": "deepseek-flash", "intent_output_cap": 256, "answer_output_cap": 1536,
                            "endpoint": "https://api.deepseek.com/chat/completions", "timeout_seconds": 60})
            calls = []
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "SYNTHETIC-only-key"}), patch("httpx.HTTPTransport", return_value=transport):
                with self.assertRaises(json.JSONDecodeError):
                    api.call(folder, "identify", [], calls)
            self.assertEqual(len(calls), 1)
            self.assertEqual((folder / "api_1_response.bin").read_bytes(), b"SYNTHETIC not JSON")
            result = demo.read(folder / "api_1_result.json")
            self.assertEqual(result["request_id"], "SYNTHETIC-id")
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["received_bytes"], 18)


if __name__ == "__main__":
    unittest.main()
