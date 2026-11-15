"""A collapsed generator is an explicit unavailable utility outcome, never a fabricated loss."""
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PACKAGE/'src'), str(PACKAGE/'scripts')]
from tabpollution.studies import broad_formal_batch as batch
from tabpollution.studies import broad_utility as utility
from tabpollution.governance.broad_batch_registry import identity_sha256


@pytest.fixture
def collapsed_outcome(monkeypatch, tmp_path):
    config = batch.load_json(PACKAGE/'tests/fixtures/v1_synthetic_config.json')
    config['utility']['paired_bootstrap_replicates'] = 20
    plan = batch.build_plan(config)
    frame = pd.DataFrame({'record_id': [f'artificial:{i}' for i in range(230)],
        'x': np.arange(230, dtype=float), 'category': ['a', 'b']*115,
        'target': ['no', 'yes']*115})
    roles = frame[['record_id']].copy()
    roles['role'] = ['train']*160+['evaluation']*50+['tune']*10+['final_test']*10
    data = dict(task='binary', features=['x', 'category'], numeric=['x'], categorical=['category'],
        target='target', train=frame.iloc[:160].copy(), evaluation=frame.iloc[160:210].copy(),
        roles=roles, roles_sha256=hashlib.sha256(roles.to_csv(index=False).encode()).hexdigest(),
        source_sha256='c'*64, intake={'dataset_id':'bank_marketing'}, group_features=['x'])
    generator_calls, task_calls = [], []
    class Generator:
        def __init__(self, name): self.name=name
        def fit(self, data, **kwargs): self.data=data.copy(); generator_calls.append(self.name)
        def sample(self, n, **kwargs):
            result=self.data.copy()
            if self.name == 'TVAE': result['target']='no'
            return result
        def get_provenance(self): return {'config':config['generators'][self.name]}
    class Model:
        def fit(self, x, y):
            self.classes_=np.array(sorted(pd.Series(y).unique())); task_calls.append(1)
        def predict_proba(self, x): return np.full((len(x),len(self.classes_)),1/len(self.classes_))
    monkeypatch.setattr(utility,'build_model',lambda *a, **k:Model())
    monkeypatch.setattr(utility.SDVGenerator,'build_metadata',staticmethod(lambda *a:{}))
    report=batch.execute_unit(data,seed=203101,plan=plan,config=config,generator_factory=Generator)
    assert len(generator_calls)==3 and len(task_calls)==80
    assert report['downstream_fit_calls']==80 and report['invalid_training_outcomes']==4
    invalid=[r for c in report['matrix']['cases'].values() for r in c['results']
             if r['status']=='invalid_training_class_support']
    assert len(invalid)==4 and all(r['arm']=='TSTR' and r['metrics'] is None for r in invalid)
    assert all(r['class_support']['missing_class_count']==1 for r in invalid)
    identity=batch.build_identity_for_unit(config,plan,'bank_marketing',203101,data,'a'*64,'b'*64)
    report.update(producer_id='artificial-worker',identity=identity,identity_sha256=identity_sha256(identity),
        package_sha256='a'*64,runtime_sha256='b'*64,generator_names=list(config['generators']))
    for name, value in [('config.json',config),('plan.json',plan),('identity.json',identity),('aggregate.json',report)]:
        (tmp_path/name).write_text(json.dumps(value,allow_nan=False),encoding='utf-8')
    spec=importlib.util.spec_from_file_location('independent_invalid_auditor',PACKAGE/'scripts/audit_broad_formal_unit.py')
    auditor=importlib.util.module_from_spec(spec);spec.loader.exec_module(auditor)
    return report, auditor, tmp_path


def run_audit(auditor, root, name='receipt.json'):
    return auditor.audit(root/'aggregate.json',root/'identity.json',root/name,package_sha='a'*64,
        runtime_sha='b'*64,config_path=root/'config.json',plan_path=root/'plan.json')


def test_independent_audit_accounts_80_fits_and_four_unavailable_without_metrics(collapsed_outcome):
    report,auditor,root=collapsed_outcome
    receipt=run_audit(auditor,root)
    assert receipt['downstream_fit_calls']==80 and receipt['invalid_training_outcomes']==4
    assert receipt['downstream_fit_budget']==84


@pytest.mark.parametrize('tamper',['fake_metric','fake_count','drop_arm','missing_support','unknown_failure'])
def test_independent_auditor_rejects_hidden_or_fabricated_outcomes(collapsed_outcome,tamper):
    report,auditor,root=collapsed_outcome
    changed=copy.deepcopy(report)
    rows=changed['matrix']['cases']['TVAE']['results']
    invalid=next(r for r in rows if r['status']=='invalid_training_class_support')
    if tamper=='fake_metric': invalid['metrics']={'log_loss':0.0}
    elif tamper=='fake_count': changed['downstream_fit_calls']=84
    elif tamper=='drop_arm': rows.remove(invalid)
    elif tamper=='missing_support': invalid.pop('class_support')
    else: invalid['status']='runtime_failure'
    (root/'aggregate.json').write_text(json.dumps(changed),encoding='utf-8')
    with pytest.raises(ValueError): run_audit(auditor,root)


def test_mixed_arm_properties_are_distinct_and_bound_to_rows(collapsed_outcome):
    report, _, _ = collapsed_outcome
    case = report["matrix"]["cases"]["TVAE"]
    props = case["arm_properties"]
    assert props["replace_0.25"] != props["augment_0.5"]
    for row in case["results"]:
        assert row["properties"] == props[row["arm"]]
    unavailable = [r for r in case["results"] if r["status"] == "invalid_training_class_support"]
    assert unavailable and all(r["properties"] == props[r["arm"]] for r in unavailable)


def test_cached_control_keeps_metrics_and_rebinds_current_arm_properties(collapsed_outcome, monkeypatch):
    report, _, _ = collapsed_outcome
    config = batch.load_json(PACKAGE/'tests/fixtures/v1_synthetic_config.json')
    config['utility']['paired_bootstrap_replicates'] = 20
    plan = batch.build_plan(config)
    # Reuse fixture's synthetic construction by checking cache behavior at evaluator level.
    frame = pd.DataFrame({'record_id':[f'id:{i}' for i in range(20)], 'x':np.arange(20,dtype=float),
                          'category':['a','b']*10, 'target':['no','yes']*10})
    generated = frame[['x','category','target']].copy()
    generated['x'] += 10.0
    calls=[]
    class Model:
        def fit(self,x,y): calls.append(1); self.classes_=np.array(sorted(pd.Series(y).unique()))
        def predict_proba(self,x): return np.full((len(x),len(self.classes_)),1/len(self.classes_))
    monkeypatch.setattr(utility,'build_model',lambda *a,**k:Model())
    cache={}
    kwargs=dict(target='target',numeric=['x'],categorical=['category'],task='binary',classes=('no','yes'),
                learners=('linear',),seed=7,bootstrap_reps=20,fitted_control_cache=cache)
    first=utility.evaluate_case(frame.iloc[:12],frame.iloc[12:],frame[['x','category','target']],**kwargs)
    before={r['arm']:r.get('metrics') for r in first['results']}
    for cached in cache.values():
        cached['row']['properties']={'sentinel':'stale-cache-property'}
    calls_before=len(calls)
    second=utility.evaluate_case(frame.iloc[:12],frame.iloc[12:],generated,**kwargs)
    reused=[r for r in second['results'] if r.get('fit_reused')]
    assert reused and len(calls)>calls_before
    assert all(r['arm']=='TRTR' or r['arm'].startswith('bootstrap_') for r in reused)
    assert all(r['metrics']==before[r['arm']] for r in reused)
    assert all(r['properties']!='sentinel' and r['properties']==second['arm_properties'][r['arm']] for r in reused)
    assert all(r['properties']==second['arm_properties'][r['arm']] for r in second['results'])


def test_make_arms_dose_rounding_preserves_legacy_and_supports_floor():
    train=pd.DataFrame({'x':[0.,1.,2.],'target':['a','b','a']})
    synthetic=pd.DataFrame({'x':[10.,11.,12.],'target':['a','b','a']})
    common=dict(target='target',features=['x'],seed=3,replacement=(.5,),augmentation=(.5,))
    legacy=utility.make_arms(train,synthetic,**common)
    floor=utility.make_arms(train,synthetic,dose_rounding='floor',**common)
    assert sum(legacy['replace_0.5'].x.isin(synthetic.x))==2
    assert sum(floor['replace_0.5'].x.isin(synthetic.x))==1
    assert len(legacy['augment_0.5'])==5
    assert len(floor['augment_0.5'])==4


def test_calculate_fit_plan_accounts_for_control_cache_across_pool_counts():
    calc=batch.calculate_fit_plan
    old=calc(3,[203101,203102,203103],3,4,[.25,.5],[.5,1.0])
    assert old=={'unit_count':9,'generator_fit_calls':27,'planned_outcomes':1188,'downstream_fit_calls':756}
    five=calc(3,[204101,204102,204103,204104,204105],3,4,[.25,.5],[.5,1.0])
    assert five=={'unit_count':15,'generator_fit_calls':45,'planned_outcomes':1980,'downstream_fit_calls':1260}
    fewer_doses=calc(3,[1],2,4,[.5],[.5])
    assert fewer_doses=={'unit_count':3,'generator_fit_calls':6,'planned_outcomes':168,'downstream_fit_calls':120}
    more_generators=calc(1,[1],4,2,[.5],[.5])
    assert more_generators['unit_count']==1 and more_generators['generator_fit_calls']==4


def test_build_plan_uses_dynamic_candidate_budget_and_keeps_v1_gate():
    c=batch.load_json(PACKAGE/'broad_utility_future_v2_candidate_NO_RUN.json')
    p=batch.build_plan(c)
    expected=batch.calculate_fit_plan(len(c['datasets']),c['seeds'],len(c['generators']),len(c['learner_names']),c['utility']['replacement_doses'],c['utility']['augmentation_doses'])
    assert p['unit_count']==expected['unit_count']==75
    assert p['generator_fit_calls']==expected['generator_fit_calls']==225
    assert p['planned_outcomes']==9900
    assert p['downstream_fit_calls']==expected['downstream_fit_calls']==6300
    old=batch.load_json(PACKAGE/'tests/fixtures/v1_synthetic_config.json')
    old['protocol_id']='broad-utility-v1-preparation'
    with pytest.raises(ValueError):
        old['utility']['replacement_doses']=[.25]
        batch.build_plan(old)


def test_auditor_rejects_unbound_or_tampered_plan_fields():
    c=batch.load_json(PACKAGE/'broad_utility_future_v2_candidate_NO_RUN.json')
    trusted=batch.build_plan(c)
    assert trusted['unit_count']==75 and trusted['downstream_fit_calls']==6300
    forged=dict(trusted); forged['jobs_sha256']='0'*64
    assert forged['jobs_sha256'] != trusted['jobs_sha256']
    forged=dict(trusted); forged['downstream_fit_calls']+=1
    assert forged['downstream_fit_calls'] != trusted['downstream_fit_calls']
    missing=dict(trusted); missing.pop('jobs_sha256')
    assert 'jobs_sha256' not in missing


def test_v2_template_keeps_unknown_source_and_role_identities_null_and_fails_closed():
    c=batch.load_json(PACKAGE/'broad_utility_future_v2_candidate_NO_RUN.json')
    assert len(c['datasets'])==15
    assert all(row.get('archive_sha256') is None for row in c['datasets'])
    assert all(row.get('role_assignment_sha256') is None for row in c['datasets'])
    with pytest.raises(PermissionError,match='source and role identities required'):
        batch.require_source_role_identities(c)

def test_v2_plan_artifact_is_plan_only_and_preserves_unresolved_hashes():
    p=batch.load_json(PACKAGE/'broad_utility_future_v2_plan_ONLY.json')
    assert p['status']=='NO-RUN_CANDIDATE_NOT_AUTHORIZED_NOT_FROZEN'
    assert p['execution_enabled'] is False and p['jobs_are_plan_only'] is True
    assert p['unit_count']==75 and p['generator_fit_calls']==225
    assert p['planned_outcomes']==9900 and p['downstream_fit_calls']==6300
    assert all(value is None for value in p['source_archive_hashes'].values())
    assert all(value is None for value in p['role_assignment_hashes'].values())
    assert p['jobs_sha256'] != p['frozen_manifest_jobs_sha256']
