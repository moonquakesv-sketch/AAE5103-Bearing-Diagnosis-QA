"""Synthetic Q01 control-flow tests; no E01 data or semantic oracle is loaded.

Run with Q01_RUN_DIR pointing to this invocation's new output directory.  Every
fixture, subprocess output and machine-readable observation stays below it.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import validate_k01 as k01
from q01_flow import ScriptedProvider, group_config, load_config, run_flow, write_record


REQUEST = {
    "predicted_class": "outer_race",
    "scenario_id": "SC-K01-OR-001",
    "question": "SYNTHETIC ALPHA TOKEN: describe the abstract card.",
    "task_types": ["inspection"],
    "explicitly_allowed_operating_context": {},
}


def answer(citation="SYN-EV-001", sufficiency="sufficient"):
    return {"answer": "SYNTHETIC ALPHA is only an abstract flow-test statement.",
            "citations": [citation], "evidence_sufficiency": sufficiency}


def card(identifier="SYN-EV-001", rank=1):
    """Build all published K01 columns using visibly abstract synthetic text."""
    evidence = {field: "SYNTHETIC " + field for field in k01.FIELDS["evidence"].split()}
    evidence.update(evidence_id=identifier, source_id="SYN-SOURCE", knowledge_id="SYN-KNOWLEDGE",
                    task_type="inspection", original_text="SYNTHETIC ALPHA original text.",
                    chinese_paraphrase="合成抽象流程测试卡片。", claim_en="SYNTHETIC ALPHA paraphrase.",
                    conditions="SYNTHETIC condition must be retained.",
                    limitations="SYNTHETIC limitation must be retained.", modality="inspection_guidance")
    source = {field: "SYNTHETIC " + field for field in k01.FIELDS["sources"].split()}
    source.update(source_id="SYN-SOURCE", original_url="https://example.invalid/synthetic",
                  local_path="", sha256="")
    evidence.update(source=source, rank=rank, score=0.9 / rank,
                    relation_paths=[[{"from_node": "SYN-A", "to_node": identifier,
                                      "direction": "forward", "edge": {"relation_id": "SYN-REL"}}]])
    return evidence


def retrieval(status="retrieved", cards=None):
    if cards is None:
        cards = [card()]
    scenario = {field: "SYNTHETIC " + field for field in k01.FIELDS["scenarios"].split()}
    scenario.update(scenario_id=REQUEST["scenario_id"], predicted_class="outer_race",
                    assumption_text="SYNTHETIC registered scenario assumption.",
                    unknown_information="SYNTHETIC unknown must remain unknown.")
    return {
        "status": status, "reason": "SYNTHETIC retriever outcome: " + status,
        "request": copy.deepcopy(REQUEST), "results": copy.deepcopy(cards),
        "returned_count": len(cards), "implementation_version": "SYNTHETIC-K02",
        "knowledge_version": "SYNTHETIC-K01", "retrieval_mode": "text",
        "configuration_id": "SYNTHETIC-CONFIG", "index_id": "SYNTHETIC-INDEX",
        "shared_config_id": "SYNTHETIC-SHARED", "config_file_sha256": "0" * 64,
        "score_semantics": "SYNTHETIC similarity only; not truth or probability",
        "budget": {"top_k": 3, "unit": "unique_complete_evidence_cards",
                   "truncation": "none", "unused_slots": 3 - len(cards)},
        "information_separation": {"demo_scenario": scenario,
                                   "unknowns": scenario["unknown_information"]},
        "notices": ["SYNTHETIC public scope boundary."],
        "claim_en_semantics": "SYNTHETIC paraphrase, not original quotation.",
    }


class FakeRetriever:
    identity = {"kind": "synthetic", "version": "synthetic-fixture-v1"}

    def __init__(self, result=None, error=None):
        self.result = result if result is not None else retrieval()
        self.error = error
        self.calls = []

    def retrieve(self, request, mode="text", top_k=3):
        self.calls.append({"request": copy.deepcopy(request), "mode": mode, "top_k": top_k})
        if self.error:
            raise RuntimeError(self.error)
        result = copy.deepcopy(self.result)
        result["retrieval_mode"] = mode
        return result


def synthetic_case(name):
    req, group, ret = copy.deepcopy(REQUEST), "TV", FakeRetriever()
    script = {"generate": [answer()], "verify": [{"verdict": "pass", "issues": []}],
              "revise": [answer()]}
    if name.endswith("_clean") and name.split("_")[0] in {"T", "G", "TV", "GV"}:
        group = name.split("_")[0]
    if "revision" in name or name == "revision_still_issues":
        script["verify"] = [{"verdict": "issues", "issues": ["SYNTHETIC issue"]},
                            {"verdict": "pass", "issues": []}]
    if name == "revision_still_issues":
        script["verify"][1] = {"verdict": "issues", "issues": ["SYNTHETIC remaining issue"]}
        script["revise"].append(answer())  # A second item must remain unconsumed.
    if name == "revision_schema_failed":
        script["revise"] = [{"answer": "SYNTHETIC malformed revision"}]
    if name == "revision_citation_failed":
        script["revise"] = [answer("SYN-EV-UNRETURNED")]
    if name in {"unknown_citation", "out_of_return_set_citation"}:
        script["generate"] = [answer("SYN-EV-999" if name == "unknown_citation" else "SYN-EV-002")]
    if name == "duplicate_citation":
        script["generate"][0]["citations"].append("SYN-EV-001")
    if name == "missing_generation_field":
        del script["generate"][0]["evidence_sufficiency"]
    if name == "extra_generation_field":
        script["generate"][0]["provider_self_reported_pass"] = True
    if name == "generation_exception":
        script["generate"] = [{"raise": "SYNTHETIC generation exception"}]
    if name == "verify_exception":
        script["verify"] = [{"raise": "SYNTHETIC verification exception"}]
    if name == "verify_missing_field":
        script["verify"] = [{"verdict": "pass"}]
    if name == "verify_invalid_pass":
        script["verify"] = [{"verdict": "pass", "issues": ["SYNTHETIC contradiction"]}]
    if name == "revision_exception":
        script["revise"] = [{"raise": "SYNTHETIC revision exception"}]
    if name == "post_revision_verify_exception":
        script["verify"][1] = {"raise": "SYNTHETIC final verification exception"}
    empties = {"valid_empty": "no_candidates", "empty_question": "empty_question",
               "no_valid_terms": "no_valid_terms", "no_vocabulary_terms": "no_vocabulary_terms",
               "no_candidates": "no_candidates", "zero_similarity": "zero_similarity",
               "knowledge_not_covered": "knowledge_not_covered"}
    if name in empties:
        ret = FakeRetriever(retrieval(empties[name], []))
    if name == "empty_question":
        req["question"] = " "
    if name == "knowledge_not_covered":
        req["scenario_id"] = "SYNTHETIC-UNKNOWN-SCENARIO"
    if name == "retrieval_exception":
        ret = FakeRetriever(error="SYNTHETIC retrieval exception")
    if name == "retrieval_service_error":
        ret = FakeRetriever(retrieval("service_error", []))
    if name == "provider_insufficient":
        group = "T"
        script["generate"] = [answer(sufficiency="insufficient")]
    if name == "extra_business_metadata":
        req["family"] = "SYNTHETIC-SECRET-METADATA"
    if name == "forbidden_business_secret":
        req["reference_answer"] = "SYNTHETIC-SECRET-ANSWER"
    if name == "missing_business_field":
        del req["task_types"]
    if name == "invalid_context":
        req["explicitly_allowed_operating_context"] = {"notes": "SYNTHETIC-SECRET-NOTES"}
    return req, group, ret, ScriptedProvider(script)


class Q01SyntheticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        run = Path(os.environ["Q01_RUN_DIR"]).resolve()
        cls.base = run / "synthetic"
        cls.expectations = json.loads((cls.base / "expected_cases.json").read_text())
        cls.cli_expectations = json.loads((cls.base / "expected_cli_cases.json").read_text())
        cls.attempt = Path(tempfile.mkdtemp(prefix="attempt_", dir=cls.base))
        cls.observations = []
        cls.cli_observations = []

    @classmethod
    def tearDownClass(cls):
        (cls.attempt / "actual_cases.json").write_text(
            json.dumps(cls.observations, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (cls.attempt / "actual_cli_cases.json").write_text(
            json.dumps(cls.cli_observations, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def cli(self, request_path, script_path, output_path, name):
        argv = [sys.executable, str(ROOT / "scripts/q01_flow.py"), "--request", str(request_path),
                "--group", "T", "--script", str(script_path), "--output", str(output_path)]
        p = subprocess.run(argv, cwd=self.attempt, text=True, capture_output=True, check=False)
        (self.attempt / (name + "_command.json")).write_text(json.dumps({
            "argv": argv, "cwd": str(self.attempt),
            "environment": {k: os.environ.get(k) for k in ("PYTHONDONTWRITEBYTECODE", "TMPDIR", "Q01_RUN_DIR")},
            "exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr,
        }, ensure_ascii=False, indent=2) + "\n")
        return p

    def test_cli_parser_failure_envelope(self):
        script_path = self.attempt / "synthetic_cli_script.json"
        script_path.write_text(json.dumps({"generate": [answer()]}))
        for expected in self.cli_expectations["cases"]:
            name = expected["case"]
            with self.subTest(case=name):
                request_path = self.attempt / (name + ".json")
                raw = (b"{SYNTHETIC not JSON" if name == "non_json_request" else
                       b'{"question":"SYNTHETIC A","question":"SYNTHETIC B"}')
                request_path.write_bytes(raw)
                output = self.attempt / (name + "_output.json")
                p = self.cli(request_path, script_path, output, name)
                record = json.loads(output.read_text())
                self.cli_observations.append({"case": name, "expected": expected, "record": record})
                self.assertEqual(p.returncode, expected["exit_code"], p.stderr)
                for key in ("status", "error_code", "publishable", "final_answer", "counts"):
                    self.assertEqual(record[key], expected[key], key)
                self.assertEqual(record["request_file_sha256"], hashlib.sha256(raw).hexdigest())
                self.assertEqual(record["metrics"]["token_usage"], "unknown")
                self.assertEqual(record["metrics"]["cost"], "not_applicable_offline")
                self.assertTrue(record["identity"]["code_sha256"])
                self.assertTrue(record["identity"]["config_file_sha256"])
                self.assertEqual(record["timings_seconds"], dict.fromkeys(expected["counts"], None))

    def test_cli_rejects_existing_output_before_reading_inputs(self):
        missing_request = self.attempt / "request_must_not_be_read.json"
        missing_script = self.attempt / "script_must_not_be_read.json"
        output = self.attempt / "cli_existing_output.json"
        output.write_bytes(b"SYNTHETIC cli output bytes must survive")
        before = output.read_bytes()
        p = self.cli(missing_request, missing_script, output, "existing_output")
        self.assertEqual(p.returncode, 2)
        self.assertEqual(json.loads(p.stderr)["status"], "output_rejected")
        self.assertEqual(output.read_bytes(), before)
        dangling = self.attempt / "cli_dangling_output.json"
        target = self.attempt / "cli_missing_target.json"
        dangling.symlink_to(target)
        p = self.cli(missing_request, missing_script, dangling, "dangling_output")
        self.assertEqual(p.returncode, 2)
        self.assertEqual(json.loads(p.stderr)["status"], "output_rejected")
        self.assertTrue(dangling.is_symlink())
        self.assertFalse(target.exists())
        dangling.unlink()

    def test_independent_state_and_call_oracle(self):
        for expected in self.expectations["cases"]:
            name = expected["case"]
            with self.subTest(case=name):
                req, group, ret, provider = synthetic_case(name)
                record = run_flow(req, group_config(group), provider, retriever=ret)
                observed = {"case": name, "expected": expected, "record": record,
                            "provider_calls": provider.calls, "retriever_calls": ret.calls}
                self.observations.append(observed)
                self.assertEqual(record["status"], expected["status"])
                self.assertEqual(record["publishable"], expected["publishable"])
                self.assertEqual(record["counts"], expected["counts"])
                if not expected["publishable"]:
                    self.assertIsNone(record["final_answer"], "A blocked draft must never be exposed as final text")
                if expected["counts"]["generate"] and name != "generation_exception":
                    self.assertIsNotNone(record["draft"], "Keep provider output for research diagnosis")
                self.assertLessEqual(record["counts"]["revise"], 1)
                self.assertLessEqual(record["counts"]["verify"], 2)
                self.assertLessEqual(record["counts"]["retrieval"], 1)
                self.assertEqual(len(provider.calls), sum(record["counts"][k] for k in ("generate", "verify", "revise")))
                self.assertEqual(len(ret.calls), record["counts"]["retrieval"])

    def test_no_hidden_checks_in_T_G(self):
        for group in ("T", "G"):
            provider = ScriptedProvider({"generate": [answer()],
                                         "verify": [{"raise": "SYNTHETIC hidden verification forbidden"}],
                                         "revise": [{"raise": "SYNTHETIC hidden rewrite forbidden"}]})
            record = run_flow(copy.deepcopy(REQUEST), group_config(group), provider, retriever=FakeRetriever())
            self.assertTrue(record["publishable"])
            self.assertEqual([c["action"] for c in provider.calls], ["generate"])
            self.assertEqual(record["status"], "completed_unverified")

    def test_shared_controls_and_global_budget(self):
        configs = {g: group_config(g) for g in ("T", "TV", "G", "GV")}
        shared = []
        for group, config in configs.items():
            self.assertEqual(config["top_k"], 3)
            ret = FakeRetriever(retrieval(cards=[card("SYN-EV-001", 1), card("SYN-EV-002", 2), card("SYN-EV-003", 3)]))
            provider = ScriptedProvider({"generate": [answer()], "verify": [{"verdict": "pass", "issues": []}]})
            record = run_flow(copy.deepcopy(REQUEST), config, provider, retriever=ret)
            self.assertEqual(ret.calls[0]["top_k"], 3)
            self.assertEqual(ret.calls[0]["mode"], "graph" if group in {"G", "GV"} else "text")
            payload = provider.calls[0]["payload"]
            encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
            for original in ret.result["results"]:
                for field in k01.FIELDS["evidence"].split():
                    self.assertIn(original[field], encoded)
                for field in k01.FIELDS["sources"].split():
                    if original["source"][field]:
                        self.assertIn(original["source"][field], encoded)
                self.assertIn(original["evidence_id"], encoded)
            self.assertEqual(record["returned_cards"], ret.result["results"],
                             "Raw full-card diagnostics must retain scores and relation paths")
            self.assertTrue(record["publishable"])
            shared.append(payload)
        self.assertEqual(shared[0], shared[1], "T/TV generation payloads differ despite shared retrieval mode")
        self.assertEqual(shared[2], shared[3], "G/GV generation payloads differ despite shared retrieval mode")
        self.assertEqual(shared[0], shared[2], "Equal returned cards must keep common generation controls across modes")

    def test_adjacent_secrets_and_retrieval_extras_do_not_enter_prompts(self):
        secret = "SYNTHETIC_ADJACENT_SECRET_6b81d3c9"
        (self.attempt / "evaluation_secrets.json").write_text(json.dumps({"reference_answer": secret}))
        ret_result = retrieval()
        ret_result.update(reference_answer=secret, scoring_rubrics=secret, audit_report=secret,
                          evaluation_directory=str(self.attempt / "evaluation_secrets.json"))
        ret_result["results"][0]["reference_answer"] = secret
        ret_result["results"][0]["source"]["audit_report"] = secret
        ret_result["information_separation"]["demo_scenario"]["review_notes"] = secret
        provider = ScriptedProvider({"generate": [answer()],
                                     "verify": [{"verdict": "issues", "issues": ["SYNTHETIC issue"]},
                                                {"verdict": "pass", "issues": []}],
                                     "revise": [answer()]})
        record = run_flow(copy.deepcopy(REQUEST), group_config("GV"), provider,
                          retriever=FakeRetriever(ret_result))
        for call in provider.calls:
            encoded = json.dumps(call["payload"], sort_keys=True)
            self.assertNotIn(secret, encoded)
            for forbidden in ("reference_answer", "scoring_rubrics", "audit_report", "evaluation_directory"):
                self.assertNotIn(forbidden, encoded)
        self.assertEqual(set(REQUEST), set(ret_result["request"]))
        self.assertTrue(record["publishable"])

    def test_metrics_are_unknown_or_measured(self):
        record = run_flow(copy.deepcopy(REQUEST), group_config("T"),
                          ScriptedProvider({"generate": [answer()]}), retriever=FakeRetriever())
        leaves = []
        def visit(value, trail=""):
            if isinstance(value, dict):
                for key, item in value.items():
                    visit(item, trail + "/" + key)
            else:
                leaves.append((trail, value))
        visit(record["metrics"])
        tokens = [(k, v) for k, v in leaves if "token" in k.lower()]
        costs = [(k, v) for k, v in leaves if "cost" in k.lower() or "fee" in k.lower()]
        elapsed = list(record["timings_seconds"].items())
        self.assertTrue(tokens, "Token accounting must be explicit")
        self.assertTrue(costs, "Cost accounting must be explicit")
        self.assertTrue(elapsed, "Elapsed time must be explicit")
        for key, value in tokens + costs:
            self.assertIn(value, ("unknown", "not_applicable", "not_applicable_offline"), key)
        for key, value in elapsed:
            self.assertTrue(value is None or value == "unknown" or
                            (type(value) in (int, float) and value > 0), (key, value))
            if record["counts"][key]:
                self.assertTrue(type(value) in (int, float) and value > 0, (key, value))
            else:
                self.assertIsNone(value, "A skipped stage must remain explicitly unmeasured")

    def test_output_is_exclusive_for_files_and_dangling_symlinks(self):
        existing = self.attempt / "existing.json"
        existing.write_bytes(b"SYNTHETIC-PRESERVE-BYTES")
        before = existing.read_bytes()
        with self.assertRaises(OSError):
            write_record(existing, {"value": "replace forbidden"})
        self.assertEqual(existing.read_bytes(), before)
        dangling = self.attempt / "dangling.json"
        dangling.symlink_to(self.attempt / "missing-target.json")
        with self.assertRaises(OSError):
            write_record(dangling, {"value": "replace forbidden"})
        self.assertTrue(dangling.is_symlink())
        self.assertFalse((self.attempt / "missing-target.json").exists())
        dangling.unlink()
        output = self.attempt / "new.json"
        write_record(output, {"synthetic": True})
        self.assertEqual(json.loads(output.read_text()), {"synthetic": True})

    def test_import_reads_no_evaluation_files(self):
        audit_script = self.attempt / "import_audit.py"
        audit_script.write_text(
            "import sys,json,pathlib\n"
            "seen=[]\n"
            "def audit(event,args):\n"
            " if event == 'open':\n"
            "  path=str(args[0]); seen.append(path)\n"
            "  if '/evaluation/' in path or '/runs/e01' in path:\n"
            "   raise RuntimeError('Forbidden evaluation read on import: '+path)\n"
            "sys.addaudithook(audit)\n"
            "sys.path.insert(0,sys.argv[1])\n"
            "import q01_flow\n"
            "print(json.dumps({'opened_paths':seen,'import_ok':True}))\n", encoding="utf-8")
        p = subprocess.run([sys.executable, str(audit_script), str(ROOT / "scripts")],
                           cwd=self.attempt, text=True, capture_output=True, check=False)
        (self.attempt / "import_audit.stdout").write_text(p.stdout)
        (self.attempt / "import_audit.stderr").write_text(p.stderr)
        (self.attempt / "import_audit_command.json").write_text(json.dumps({
            "argv": [sys.executable, str(audit_script), str(ROOT / "scripts")],
            "cwd": str(self.attempt),
            "environment": {k: os.environ.get(k) for k in ("PYTHONDONTWRITEBYTECODE", "TMPDIR", "Q01_RUN_DIR")},
            "exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr,
        }, ensure_ascii=False, indent=2) + "\n")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertTrue(json.loads(p.stdout)["import_ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
