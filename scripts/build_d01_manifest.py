"""Materialize the D01 selection from the persisted official-file audit."""
from pathlib import Path
import csv,json
ROOT=Path(__file__).resolve().parents[1]
audit=json.loads((ROOT/'runs/d01/source_audit.json').read_text())
rows=[]
for r in audit['records']:
    normal=r['true_label']=='normal'; hp=r['load_hp']; n=int(r['file'].split('.')[0])
    rpm_key=f'X{n:03d}RPM'; actual=r['arrays'].get(rpm_key,{}).get('scalar','')
    ignored=[k for k in r['arrays'] if k.endswith('_time') and k!=r['channel']]
    row=dict(record_id=r['record_id'],official_name=r['official_name'],source_url=r['source_url'],
       metadata_url='https://engineering.case.edu/bearingdatacenter/'+('normal-baseline-data' if normal else '12k-drive-end-bearing-fault-data'),
       file_name=r['file'],raw_path=r['raw_path'],sha256=r['sha256'],channel=r['channel'],
       channel_status='verified_scipy_loadmat',original_fs_hz=48000 if normal else 12000,
       fs_status='literature_supported_not_official_per_file' if normal else 'official_12k_DE_table',
       fs_evidence_url='https://doi.org/10.36001/phme.2022.v7i1.3329' if normal else 'https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data',
       fs_evidence_locator='sections_4.1.1_4.2_pdf_pages_6_7' if normal else 'page_title',
       load_hp=hp,rpm=actual if actual!='' else [1797,1772,1750,1730][hp],
       rpm_basis='MAT_'+rpm_key if actual!='' else 'official_table_approximate',
       rpm_table_approx=[1797,1772,1750,1730][hp],rpm_mat=actual,
       rpm_missing_reason='' if actual!='' else 'no_matching_RPM_variable_ans_not_used',
       fault_size=0 if normal else .007,fault_size_unit='inch_diameter',fault_depth_in='' if normal else .011,
       fault_position='6:00' if r['true_label']=='outer_race' else 'not_applicable',
       load_zone_relation='centered_per_download_table_prose_conflicts' if r['true_label']=='outer_race' else 'not_applicable',
       true_label=r['true_label'],fault_bearing='none' if normal else 'drive_end',sensor_location='drive_end_housing',
       bearing_model='6205-2RS JEM SKF',group_id=r['record_id'],
       source_family_id='cwru_normal' if normal else 'cwru_DE_007_'+r['true_label']+('_6oclock' if r['true_label']=='outer_race' else ''),
       physical_specimen_id='',specimen_status='not_published_in_reviewed_official_sources',
       split='train' if hp in (0,1) else 'validation' if hp==2 else 'test',
       n_samples=r['n_samples'],signal_sha256_float64=r['arrays'][r['channel']]['sha256_float64'],
       ignored_time_keys=';'.join(ignored),selection_status='selected_for_D02_pilot',
       checked_date='2026-10-21',notes='normal_48k_is_documented_project_choice_with_literature_support' if normal else '')
    if n==99: row['notes']+=';X098_arrays_duplicate_98_mat_excluded_use_X099_only'
    rows.append(row)
with (ROOT/'data/records.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(f'wrote {len(rows)} records to data/records.csv')
