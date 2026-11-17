"""K01 mapping and declared-interface regressions. No classifier imports."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import validate_k01 as k

class K01ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables=k.load(k.ROOT/'knowledge/k01')
        cls.interface=json.loads((k.ROOT/'knowledge/k01/interface.json').read_text())

    def test_current_tables_sources_and_interface(self):
        r=k.validate(self.tables,interface=self.interface)
        self.assertEqual(r['counts'],dict(sources=4,scenarios=1,evidence=11,entities=29,relations=29))
        self.assertEqual(r['task_coverage_counts'],dict(explanation=3,inspection=3,preparation=5))
        self.assertTrue(r['interface_checked'])

    def test_existing_thirteen_structural_rejections(self):
        self.assertEqual(len(k.counterexamples(self.tables)),13)

    def test_legal_but_wrong_mirrors_and_mechanism(self):
        for node in ['EV-K01-001','KN-K01-001','MECH-K01-FB']:
            with self.subTest(node=node):
                fixture=copy.deepcopy(self.tables)
                next(e for e in fixture['entities'] if e['entity_id']==node)['content_ref']='evidence.csv#EV-K01-003'
                with self.assertRaisesRegex(k.Invalid,'content_ref mapping'):
                    k.validate(fixture,check_files=False)

    def test_interface_ground_truth_conflict(self):
        fixture=copy.deepcopy(self.interface)
        fixture['allowed_query_inputs'].append('ground_truth')
        with self.assertRaisesRegex(k.Invalid,'allowed_query_inputs contract mismatch'):
            k.validate(self.tables,check_files=False,interface=fixture)

    def test_interface_lists_missing_extra_duplicate_and_wrong_type(self):
        for field in ['tables','allowed_query_inputs','forbidden_input_categories']:
            values=self.interface[field]
            for label,changed in [('missing',values[1:]),('extra',values+['unregistered']),
                                   ('duplicate',values+[values[0]]),('string','|'.join(values)),
                                   ('non_string',values+[{}])]:
                with self.subTest(field=field,mutation=label):
                    fixture=copy.deepcopy(self.interface);fixture[field]=changed
                    with self.assertRaises(k.Invalid):k.validate_interface(fixture)

    def test_interface_version_schema_flags_scope_and_authorship(self):
        mutations={'knowledge_version':'20261023_rev010','schema_version':'k01.v2',
            'implementation_status':'retrieval_complete','assumptions_are_not_evidence':False,
            'audit_material_is_retrieval_input':True,'scope':'aviation release approved',
            'claim_en_semantics':'Human-authored English paraphrase, not a verbatim manufacturer quotation.',
            'separator_for_id_lists':',','encoding':'ASCII','pdf_file_page_base':0}
        for field,value in mutations.items():
            with self.subTest(field=field):
                fixture=copy.deepcopy(self.interface);fixture[field]=value
                with self.assertRaises(k.Invalid):k.validate_interface(fixture)

    def test_bool_is_not_integer_page_base_or_integer_flag(self):
        for field,value in [('pdf_file_page_base',True),('assumptions_are_not_evidence',1)]:
            with self.subTest(field=field):
                fixture=copy.deepcopy(self.interface);fixture[field]=value
                with self.assertRaises(k.Invalid):k.validate_interface(fixture)

    def test_interface_missing_unknown_fields_and_empty_policy(self):
        for mutation in ['missing','extra','empty_policy']:
            with self.subTest(mutation=mutation):
                fixture=copy.deepcopy(self.interface)
                if mutation=='missing':del fixture['null_policy']
                elif mutation=='extra':fixture['ground_truth']='outer_race'
                else:fixture['retrieval_corpus_policy']=' '
                with self.assertRaises(k.Invalid):k.validate_interface(fixture)

    def test_table_only_mode_reports_no_interface_check(self):
        self.assertFalse(k.validate(self.tables,check_files=False)['interface_checked'])

    def test_cli_uses_selected_folder_interface_and_exclusive_output(self):
        with tempfile.TemporaryDirectory(prefix='cli_',dir=k.ROOT/'runs/k01_fix/rev011') as folder:
            folder=Path(folder);knowledge=folder/'knowledge'
            shutil.copytree(k.ROOT/'knowledge/k01',knowledge)
            output=folder/'result.json'
            command=[sys.executable,str(k.ROOT/'scripts/validate_k01.py'),'--knowledge',str(knowledge),'--output',str(output)]
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
            good=subprocess.run(command,env=env,capture_output=True,text=True)
            self.assertEqual(good.returncode,0,good.stderr)
            original_bytes=output.read_bytes()
            duplicate=subprocess.run(command,env=env,capture_output=True,text=True)
            self.assertNotEqual(duplicate.returncode,0)
            self.assertEqual(output.read_bytes(),original_bytes)
            interface=copy.deepcopy(self.interface);interface['allowed_query_inputs'].append('ground_truth')
            (knowledge/'interface.json').write_text(json.dumps(interface))
            command[-1]=str(folder/'bad.json')
            bad=subprocess.run(command,env=env,capture_output=True,text=True)
            self.assertNotEqual(bad.returncode,0)
            self.assertIn('allowed_query_inputs contract mismatch',bad.stderr)
            self.assertFalse((folder/'bad.json').exists())

    def test_semantic_falsehood_remains_a_manual_review_boundary(self):
        fixture=copy.deepcopy(self.tables)
        claim='Every stationary bearing exposed to external vibration necessarily develops false brinelling.'
        fixture['evidence'][1]['claim_en']=claim
        node=fixture['evidence'][1]['knowledge_id']
        next(e for e in fixture['entities'] if e['entity_id']==node)['label']=claim
        self.assertEqual(k.validate(fixture,check_files=False)['status'],'PASS')

if __name__=='__main__':unittest.main(verbosity=2)
