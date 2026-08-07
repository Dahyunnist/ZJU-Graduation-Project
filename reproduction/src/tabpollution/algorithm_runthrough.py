"""End-to-end algorithm runthrough used for the fourth execution round.

The module intentionally operates on deterministic CPU-sized subsets.  Every
run is labelled ``runthrough`` and is kept out of the formal benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
import hashlib
import json
import os
import platform
import time

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, f1_score, log_loss, roc_auc_score,
                             roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tabpollution.detectors import C2STDetector, Char3GramDetector, DeepTextDetector
from tabpollution.detectors.base import PROVENANCE_COLUMNS, feature_frame
from tabpollution.generators.sdv_adapter import create_generator
from tabpollution.quantification import ScoreQuantifier, available_quantifiers
from tabpollution.utils import sha256_file, write_json
from tabpollution.valuation import data_oob, exact_knn_shapley, knn_shapley


RATES = (0., .05, .10, .25, .50, .75, 1.)
ALGORITHM_IDS = ("D-LR", "D-XGB", "D-3G", "D-FT", "D-TT", "D-DW", "D-DWTA",
                 "Q-CC", "Q-PCC", "Q-ACC", "Q-PACC", "Q-EMQ", "Q-HDy", "Q-DyS", "Q-MS",
                 "V-CURVE-LR", "V-CURVE-XGB", "V-KNN", "V-OOB")


def utc_now() -> str: return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Path): return str(value)
    return value


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    total = 0.
    for lo, hi in zip(np.linspace(0, 1, bins+1)[:-1], np.linspace(0, 1, bins+1)[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any(): total += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def classification_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    y, score = np.asarray(y, int), np.asarray(score, float)
    pred = (score >= .5).astype(int); fpr, tpr, _ = roc_curve(y, score)
    return {"auroc": float(roc_auc_score(y, score)), "auprc": float(average_precision_score(y, score)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "f1": float(f1_score(y, pred)),
            "tpr_at_fpr_005": float(np.interp(.05, fpr, tpr)),
            "brier": float(brier_score_loss(y, score)), "ece": _ece(y, score)}


def _hashes(root: Path) -> dict[str, str]:
    rels = ["data/processed/adult/adult_clean.csv", "data/processed/credit/credit_clean.csv",
            "data/splits/benchmark_v1/adult/seed_2026.csv", "data/splits/benchmark_v1/credit/seed_2026.csv",
            "runs/c2-smoke-adult-gaussiancopula-s42-a2/smoke_summary.json",
            "runs/c3-smoke-adult-gaussiancopula-s42/smoke_summary.json",
            "runs/c2-pilot-adult-gaussiancopula-s2026-20260714T181119905473Z/pilot_summary.json"]
    return {r: sha256_file(root/r) for r in rels if (root/r).exists()}


@dataclass
class Context:
    root: Path
    reports: Path
    seed: int = 2026

    def __post_init__(self) -> None:
        self.runs = self.root / "runs"; self.out = self.reports / "algorithm_runthrough"
        self.out.mkdir(parents=True, exist_ok=True)
        self.status: dict[str, dict[str, Any]] = {a: {"algorithm_id": a, "status": "planned"} for a in ALGORITHM_IDS}
        self.run_index: list[dict[str, Any]] = []

    def run(self, algorithm: str, body: Callable[[Path], tuple[dict[str, Any], pd.DataFrame | None]],
            config: dict[str, Any]) -> dict[str, Any]:
        run_id = f"runthrough-{algorithm.lower()}-{datetime.now().strftime('%Y%m%dT%H%M%S%f')}"
        directory = self.runs / run_id; directory.mkdir(parents=True, exist_ok=False)
        started = time.perf_counter(); (directory/"stdout.log").write_text("", encoding="utf-8"); (directory/"stderr.log").write_text("", encoding="utf-8")
        config = {**config, "algorithm_id": algorithm, "run_type": "runthrough", "seed": self.seed}
        (directory/"config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        write_json({"python": platform.python_version(), "platform": platform.platform(), "cpu_count": os.cpu_count()}, directory/"environment.json")
        try:
            metrics, table = body(directory)
            if table is not None: table.to_csv(directory/"predictions.csv", index=False)
            elapsed = time.perf_counter()-started
            write_json(metrics, directory/"metrics.json"); write_json({"total_seconds": elapsed}, directory/"timing.json")
            provenance = metrics.pop("_provenance", {}) if "_provenance" in metrics else {}
            write_json(provenance, directory/"provenance.json")
            write_json({"run_id": run_id, "algorithm_id": algorithm, "run_type": "runthrough",
                        "status": "runthrough_passed", "created_at": utc_now()}, directory/"run_manifest.json")
            artifacts = []
            for p in sorted(directory.rglob("*")):
                if p.is_file(): artifacts.append({"path": p.relative_to(directory).as_posix(), "bytes": p.stat().st_size, "sha256": sha256_file(p)})
            write_json({"artifacts": artifacts}, directory/"artifacts_manifest.json")
            write_json({"status": "runthrough_passed", "run_type": "runthrough"}, directory/"status.json")
            (directory/"COMPLETE").write_text("complete\n", encoding="utf-8")
            self.status[algorithm].update(status="runthrough_passed", run_id=run_id, metrics=metrics, seconds=elapsed)
            self.run_index.append({"algorithm_id": algorithm, "run_id": run_id, "status": "runthrough_passed", "seconds": elapsed, "path": str(directory)})
            return {"run_id": run_id, "metrics": metrics, "directory": directory}
        except Exception as exc:
            elapsed = time.perf_counter()-started
            (directory/"stderr.log").write_text(repr(exc)+"\n", encoding="utf-8")
            write_json({"status": "failed", "error": repr(exc), "run_type": "runthrough"}, directory/"status.json")
            write_json({"total_seconds": elapsed}, directory/"timing.json")
            self.status[algorithm].update(status="failed", error=repr(exc), run_id=run_id)
            self.run_index.append({"algorithm_id": algorithm, "run_id": run_id, "status": "failed", "seconds": elapsed, "path": str(directory)})
            raise


def _real_split(root: Path, dataset: str, split: str) -> pd.DataFrame:
    clean = pd.read_csv(root/f"data/processed/{dataset}/{dataset}_clean.csv")
    manifest = pd.read_csv(root/f"data/splits/benchmark_v1/{dataset}/seed_2026.csv")
    ids = manifest.loc[manifest.split == split, "row_id"]
    return clean.set_index("row_id").loc[ids].reset_index()


def _adult_synth(root: Path, pool: str, pilot: bool = False) -> pd.DataFrame:
    run = ("c2-pilot-adult-gaussiancopula-s2026-20260714T181119905473Z" if pilot else
           "c2-smoke-adult-gaussiancopula-s42-a2")
    return pd.read_csv(root/f"runs/{run}/pools/{pool}.csv")


def balanced(real: pd.DataFrame, synth: pd.DataFrame, n: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    n = min(n, len(real), len(synth)); a = real.sample(n, random_state=seed); b = synth.sample(n, random_state=seed)
    return pd.concat([a, b], ignore_index=True, sort=False), np.r_[np.zeros(n, int), np.ones(n, int)]


def prepare_three_tables(ctx: Context) -> dict[str, dict[str, pd.DataFrame]]:
    root = ctx.root; processed = root/"data/processed/abalone"; processed.mkdir(parents=True, exist_ok=True)
    raw = root/"data/raw/abalone/extracted/abalone.data"
    cols = ["sex", "length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight", "rings"]
    abalone = pd.read_csv(raw, names=cols)
    abalone.insert(0, "row_id", [f"abalone:{hashlib.sha256(('|'.join(map(str,r))).encode()).hexdigest()[:24]}:{i:04d}" for i, r in enumerate(abalone.astype(str).values)])
    abalone.to_csv(processed/"abalone_clean.csv", index=False)
    rng = np.random.default_rng(ctx.seed); order = rng.permutation(len(abalone)); sizes = [2506, 626, 418, 627]
    labels = np.empty(len(abalone), object); start=0
    for name, size in zip(("R_source_train","R_detector_train","R_detector_val","R_final_test"), sizes): labels[order[start:start+size]]=name; start+=size
    pd.DataFrame({"row_id": abalone.row_id, "split": labels, "target": abalone.rings}).to_csv(processed/"seed_2026.csv", index=False)
    tables: dict[str, dict[str, pd.DataFrame]] = {}
    for name in ("adult", "credit"):
        tables[name] = {s: _real_split(root, name, s) for s in ("R_source_train","R_detector_train","R_detector_val","R_final_test")}
    tables["abalone"] = {s: abalone.loc[labels == s].copy() for s in ("R_source_train","R_detector_train","R_detector_val","R_final_test")}
    tables["adult"]["S_detector_train"] = _adult_synth(root,"S_detector_train")
    tables["adult"]["S_detector_val"] = _adult_synth(root,"S_detector_val")
    tables["adult"]["S_final_test"] = _adult_synth(root,"S_final_test")
    synth_dir = root/"data/runthrough_synthetic"; synth_dir.mkdir(parents=True, exist_ok=True)
    for idx, name in enumerate(("credit", "abalone")):
        model_file=synth_dir/f"{name}_gaussiancopula.pkl"
        feature = feature_frame(tables[name]["R_source_train"].sample(min(2500,len(tables[name]["R_source_train"])), random_state=ctx.seed))
        gen=create_generator("GaussianCopula", {"enforce_min_max_values": True, "enforce_rounding": True})
        meta=gen.build_metadata(feature); gen.fit(feature, meta, ctx.seed+idx)
        gen.save(model_file)
        for j,(rp,sp) in enumerate((("R_detector_train","S_detector_train"),("R_detector_val","S_detector_val"),("R_final_test","S_final_test"))):
            n=min(1200,len(tables[name][rp])); frame=gen.sample(n, ctx.seed+100+idx*10+j, sp)
            frame.insert(0,"synth_row_id",[f"syn:{name}:GaussianCopula:{sp}:{i:06d}" for i in range(n)])
            frame["dataset_id"]=name; frame["generator_name"]="GaussianCopula"; frame["pool_name"]=sp
            frame.to_csv(synth_dir/f"{name}_{sp}.csv",index=False); tables[name][sp]=frame
    write_json({"source_url":"https://archive.ics.uci.edu/static/public/1/abalone.zip","license":"CC BY 4.0",
                "downloaded_at":"2026-07-15","sha256":sha256_file(root/"data/raw/abalone/abalone.zip"),
                "rows":len(abalone),"columns":len(cols),"role":"three-table mini cross-table runthrough only"}, ctx.reports/"abalone_runthrough_data_card.json")
    return tables


def run_classical(ctx: Context) -> tuple[dict[str, Any], dict[str, Any]]:
    train,ytr=balanced(_real_split(ctx.root,"adult","R_detector_train"),_adult_synth(ctx.root,"S_detector_train"),2000,ctx.seed)
    val,yv=balanced(_real_split(ctx.root,"adult","R_detector_val"),_adult_synth(ctx.root,"S_detector_val"),1000,ctx.seed+1)
    test,yt=balanced(_real_split(ctx.root,"adult","R_final_test"),_adult_synth(ctx.root,"S_final_test"),1500,ctx.seed+2)
    results={}; models={}
    for aid,kind in (("D-LR","lr"),("D-XGB","xgb")):
        def body(d: Path, kind=kind):
            model=C2STDetector(kind,ctx.seed).fit(train,ytr,val,yv); score=model.predict_score(test); model.save(d/"model.pkl")
            loaded=C2STDetector.load(d/"model.pkl"); reload_delta=float(np.max(np.abs(score-loaded.predict_score(test))))
            permutation_aurocs=[]
            for rep in range(5):
                rng=np.random.default_rng(ctx.seed+100+rep); perm=rng.permutation(ytr)
                sanity=C2STDetector(kind,ctx.seed+100+rep).fit(train,perm,val,rng.permutation(yv)).predict_score(test)
                permutation_aurocs.append(float(roc_auc_score(yt,sanity)))
            metrics=classification_metrics(yt,score); metrics.update(reload_max_delta=reload_delta,
                label_permutation_auroc=float(np.mean(permutation_aurocs)),label_permutation_aurocs=permutation_aurocs)
            if kind=="lr": metrics["pmse"]=float(np.mean((score-.5)**2))
            metrics["_provenance"]=model.get_provenance()
            pred=pd.DataFrame({"record_id":[*(test.row_id.fillna("").astype(str) if "row_id" in test else [""]*len(test))],"source_label":yt,"raw_score":score,"probability":score,"prediction":score>=.5})
            return metrics,pred
        results[aid]=ctx.run(aid,body,{"dataset":"adult","generator":"GaussianCopula","protocol":"P1 smoke","n_train":len(train),"n_test":len(test)})
        models[aid]=C2STDetector.load(results[aid]["directory"]/"model.pkl")
    return results,models


def format_artifact_diagnostic(project_root: Path) -> dict[str, Any]:
    """Train a detector using row-format summaries only, never feature values."""
    root=project_root.resolve()
    train,ytr=balanced(_real_split(root,"adult","R_detector_train"),_adult_synth(root,"S_detector_train"),2000,2026)
    test,yt=balanced(_real_split(root,"adult","R_final_test"),_adult_synth(root,"S_final_test"),1500,2028)
    def metadata(frame:pd.DataFrame)->np.ndarray:
        x=feature_frame(frame)
        missing=x.isna().sum(axis=1).to_numpy(); empty=x.astype("string").fillna("").apply(lambda r:sum(not str(v).strip() for v in r),axis=1).to_numpy()
        lengths=x.astype("string").fillna("<NA>").apply(lambda r:sum(len(str(v)) for v in r),axis=1).to_numpy()
        return np.c_[missing,empty,lengths]
    model=Pipeline([("scale",StandardScaler()),("lr",LogisticRegression(max_iter=300,random_state=2026))]).fit(metadata(train),ytr)
    score=model.predict_proba(metadata(test))[:,1]; auroc=float(roc_auc_score(yt,score))
    result={"dataset":"adult","generator":"GaussianCopula","features":["missing_count","empty_string_count","total_serialized_value_length"],
            "auroc":auroc,"warning":auroc>=.65,"interpretation":"High values indicate row-format/length artifacts; main detector results remain runthrough only."}
    write_json(result,root/"reports/algorithm_runthrough/format_artifact_diagnostic.json"); return result


def _fold_frames(tables: dict[str, dict[str,pd.DataFrame]], test_table: str, n: int) -> tuple[pd.DataFrame,np.ndarray,pd.DataFrame,np.ndarray,np.ndarray]:
    train_frames=[]; labels=[]; table_labels=[]
    for ti,name in enumerate([x for x in tables if x!=test_table]):
        f,y=balanced(tables[name]["R_detector_train"],tables[name]["S_detector_train"],n,2026+ti)
        f["table_id"]=name; train_frames.append(f); labels.extend(y); table_labels.extend([ti]*len(y))
    test,ytest=balanced(tables[test_table]["R_final_test"],tables[test_table]["S_final_test"],n,2090)
    test["table_id"]=test_table
    return pd.concat(train_frames,ignore_index=True,sort=False),np.asarray(labels),test,ytest,np.asarray(table_labels)


def run_cross_table(ctx: Context, tables: dict[str,dict[str,pd.DataFrame]]) -> dict[str,Any]:
    summaries=[]; output={}
    # D-3G on all leave-one-table-out folds.
    for fold,test_table in enumerate(tables):
        train,ytr,test,yt,tl=_fold_frames(tables,test_table,220)
        def body(d:Path,train=train,ytr=ytr,test=test,yt=yt,test_table=test_table):
            model=Char3GramDetector(ctx.seed+fold).fit(train,ytr); score=model.predict_score(test); shuffled=model.predict_score(test,shuffle=True); model.save(d/"model.pkl")
            metrics=classification_metrics(yt,score); metrics["shuffled_auroc"]=float(roc_auc_score(yt,shuffled)); metrics["_provenance"]=model.get_provenance()
            return metrics,pd.DataFrame({"source_label":yt,"raw_score":score,"probability":score,"table":test_table,"fold":fold})
        result=ctx.run("D-3G",body,{"protocol":"three-table mini cross-table runthrough","test_table":test_table,"fold":fold})
        summaries.append({"algorithm_id":"D-3G","test_table":test_table,"fold":fold,"run_id":result["run_id"],**result["metrics"]}); output.setdefault("D-3G",[]).append(result)
    # Keep the status pointed at all three runs.
    ctx.status["D-3G"]["run_ids"]=[r["run_id"] for r in output["D-3G"]]
    train,ytr,test,yt,tl=_fold_frames(tables,"abalone",80)
    for aid,mode in (("D-FT","flat"),("D-TT","table"),("D-DW","datum"),("D-DWTA","datum_ta")):
        def body(d:Path,mode=mode,aid=aid):
            model=DeepTextDetector(mode,ctx.seed,dim=16,heads=2,layers=1,max_len=128,max_datum=24,max_columns=48,epochs=2,batch_size=32)
            model.fit(train,ytr,table_labels=tl); score=model.predict_score(test); model.save(d/"model.pt"); loaded=DeepTextDetector.load(d/"model.pt")
            reload_delta=float(np.max(np.abs(score-loaded.predict_score(test))))
            metrics=classification_metrics(yt,score); metrics.update(reload_max_delta=reload_delta,parameter_count=sum(p.numel() for p in model.model.parameters()),truncation_rate=model.truncation_rate)
            if mode in {"datum","datum_ta"}: metrics["column_permutation_max_delta"]=model.permutation_max_delta(test.head(20),4)
            metrics["_provenance"]=model.get_provenance()
            pd.DataFrame(model.loss_history).to_csv(d/"training_curve.csv",index=False)
            return metrics,pd.DataFrame({"source_label":yt,"raw_score":score,"probability":score,"table":"abalone","fold":0})
        result=ctx.run(aid,body,{"protocol":"three-table mini cross-table runthrough","train_tables":["adult","credit"],"test_table":"abalone","device":"cpu","tiny":True})
        summaries.append({"algorithm_id":aid,"test_table":"abalone","fold":0,"run_id":result["run_id"],**result["metrics"]}); output[aid]=result
    pd.DataFrame(summaries).to_csv(ctx.out/"cross_table_mini_summary.csv",index=False)
    dw=pd.DataFrame([r for r in summaries if r["algorithm_id"] in {"D-DW","D-DWTA"}]); dw.to_csv(ctx.out/"dw_dwta_ablation.csv",index=False)
    return output


def _bag_frames(ctx:Context) -> tuple[list[dict[str,Any]],pd.DataFrame,dict[str,pd.Series]]:
    run=ctx.root/"runs/c3-smoke-adult-gaussiancopula-s42"; manifest=json.loads((run/"bags_manifest.json").read_text(encoding="utf-8"))["bags"]
    members=pd.read_csv(run/"bag_members.csv"); real=pd.read_csv(ctx.root/"data/processed/adult/adult_clean.csv").set_index("row_id")
    pools={p:pd.read_csv(ctx.root/f"runs/c2-smoke-adult-gaussiancopula-s42-a2/pools/{p}.csv") for p in ("S_detector_val","S_final_test")}
    synth=pd.concat(pools.values(),ignore_index=True).set_index("synth_row_id")
    lookup={**{i:r for i,r in real.iterrows()},**{i:r for i,r in synth.iterrows()}}
    return manifest,members,lookup


def run_quantification(ctx:Context,xgb:C2STDetector) -> dict[str,Any]:
    manifest,members,lookup=_bag_frames(ctx); rows=[]
    score_cache={}
    def get_scores(bag_id:str)->np.ndarray:
        if bag_id not in score_cache:
            ids=members.loc[members.bag_id==bag_id].sort_values("position").record_id.astype(str)
            frame=pd.DataFrame([lookup[i] for i in ids]); score_cache[bag_id]=xgb.predict_score(frame)
        return score_cache[bag_id]
    cal=[m for m in manifest if m["stage"]=="calibration"]
    cscore=np.concatenate([get_scores(m["bag_id"]) for m in cal]); cy=np.concatenate([members.loc[members.bag_id==m["bag_id"]].sort_values("position").source_type.eq("synthetic").astype(int).values for m in cal])
    outputs={}
    for method in available_quantifiers():
        aid={"median_sweep":"Q-MS","hdy":"Q-HDy","dys":"Q-DyS"}.get(method,"Q-"+method.upper())
        def body(d:Path,method=method,aid=aid):
            q=ScoreQuantifier(method).fit(cscore,cy); estimates=[]
            for meta in [m for m in manifest if m["stage"]=="test"]:
                est=q.predict_prevalence(get_scores(meta["bag_id"])); estimates.append({"algorithm_id":aid,"bag_id":meta["bag_id"],"true_proportion":meta["actual_proportion"],**{k:v for k,v in est.items() if k!="diagnostics"}})
            table=pd.DataFrame(estimates); table.to_csv(d/"estimates.csv",index=False); q.save(d/"model.json")
            err=table.clipped-table.true_proportion
            metrics={"mae":float(np.abs(err).mean()),"rmse":float(np.sqrt(np.mean(err**2))),"bias":float(err.mean()),"max_absolute_error":float(np.abs(err).max()),"out_of_range_rate":float(table.out_of_range.mean()),"bags":len(table),"score_source":"D-XGB shared frozen scores","_provenance":{"implementation":"paper_aligned_local; QuaPy 0.2.0 installed for provenance/cross-check","method":method}}
            return metrics,table
        outputs[aid]=ctx.run(aid,body,{"dataset":"adult","generator":"GaussianCopula","bag_size":1000,"test_bags_per_rate":3,"shared_score":"D-XGB"})
        rows.extend(pd.read_csv(outputs[aid]["directory"]/"estimates.csv").to_dict("records"))
    estimates=pd.DataFrame(rows); estimates.to_csv(ctx.out/"quantification_bag_estimates.csv",index=False)
    estimates.groupby("algorithm_id").apply(lambda d:pd.Series({"mae":np.mean(abs(d.clipped-d.true_proportion)),"rmse":np.sqrt(np.mean((d.clipped-d.true_proportion)**2)),"bias":np.mean(d.clipped-d.true_proportion),"max_absolute_error":np.max(abs(d.clipped-d.true_proportion)),"out_of_range_rate":np.mean(d.out_of_range)}),include_groups=False).reset_index().to_csv(ctx.out/"quantification_summary.csv",index=False)
    return outputs


def _downstream_pipeline(frame:pd.DataFrame,kind:str,seed:int)->Pipeline:
    x=feature_frame(frame); numeric=x.select_dtypes(include=["number","bool"]).columns.tolist(); categorical=[c for c in x if c not in numeric]
    prep=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),numeric),("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),categorical)])
    if kind=="lr": est=LogisticRegression(max_iter=400,random_state=seed)
    else:
        from xgboost import XGBClassifier
        est=XGBClassifier(n_estimators=50,max_depth=4,learning_rate=.1,n_jobs=2,tree_method="hist",eval_metric="logloss",random_state=seed)
    return Pipeline([("prep",prep),("model",est)])


def run_utility(ctx:Context)->dict[str,Any]:
    base=ctx.root/"runs/c3-smoke-adult-gaussiancopula-s42/contamination"; test=_real_split(ctx.root,"adult","R_final_test").sample(2000,random_state=ctx.seed)
    ytest=test.income.eq(">50K").astype(int).values; results={}; all_rows=[]
    name_map={"real_append_bootstrap_control":"real_append"}
    for aid,kind in (("V-CURVE-LR","lr"),("V-CURVE-XGB","xgb")):
        def body(d:Path,kind=kind,aid=aid):
            rows=[]
            for condition in ("real_only","real_append_bootstrap_control","synthetic_append","synthetic_replace"):
                disk=name_map.get(condition,condition)
                for p in RATES:
                    frame=pd.read_csv(base/f"{disk}_p{int(round(p*100)):03d}.csv").sample(n=min(3500,len(pd.read_csv(base/f'{disk}_p{int(round(p*100)):03d}.csv',usecols=['income']))),random_state=ctx.seed)
                    y=frame.income.eq(">50K").astype(int).values; model=_downstream_pipeline(frame.drop(columns=["income"]),kind,ctx.seed); start=time.perf_counter(); model.fit(feature_frame(frame.drop(columns=["income"])),y); train_s=time.perf_counter()-start
                    score=model.predict_proba(feature_frame(test.drop(columns=["income"])))[:,1]; pred=score>=.5
                    rows.append({"algorithm_id":aid,"model":kind,"condition":condition,"proportion":p,"n_train":len(frame),"auroc":roc_auc_score(ytest,score),"auprc":average_precision_score(ytest,score),"f1":f1_score(ytest,pred),"balanced_accuracy":balanced_accuracy_score(ytest,pred),"log_loss":log_loss(ytest,score),"train_seconds":train_s})
            table=pd.DataFrame(rows); baseline=table.loc[table.condition=="real_only"].set_index("proportion").auroc; table["delta_auroc_vs_real_only"]=[r.auroc-baseline[r.proportion] for r in table.itertuples()]
            table.to_csv(d/"utility_curve.csv",index=False); return {"conditions":4,"rates":7,"fits":len(table),"mean_auroc":float(table.auroc.mean()),"_provenance":{"implementation":"local_sklearn_xgboost","test":"pure R_final_test"}},table
        results[aid]=ctx.run(aid,body,{"dataset":"adult","generator":"GaussianCopula","conditions":4,"rates":list(RATES),"test":"R_final_test"})
        all_rows.extend(pd.read_csv(results[aid]["directory"]/"utility_curve.csv").to_dict("records"))
    pd.DataFrame(all_rows).to_csv(ctx.out/"utility_curve.csv",index=False)
    try:
        import matplotlib.pyplot as plt
        df=pd.DataFrame(all_rows); fig,axes=plt.subplots(1,2,figsize=(11,4))
        for ax,(model,g) in zip(axes,df.groupby("model")):
            for condition,h in g.groupby("condition"): ax.plot(h.proportion,h.auroc,marker="o",label=condition)
            ax.set(title=model.upper(),xlabel="synthetic proportion",ylabel="AUROC"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(ctx.out/"utility_curve.png",dpi=160); plt.close(fig)
    except Exception: pass
    return results


def _valuation_matrix(ctx:Context,n:int=800):
    frame=pd.read_csv(ctx.root/"runs/c3-smoke-adult-gaussiancopula-s42/contamination/synthetic_replace_p025.csv").sample(n=n,random_state=ctx.seed).reset_index(drop=True)
    y=frame.income.eq(">50K").astype(int).values; source=frame.source_type.astype(str).values
    xraw=feature_frame(frame.drop(columns=["income"])); numeric=xraw.select_dtypes(include=["number","bool"]).columns.tolist(); categorical=[c for c in xraw if c not in numeric]
    prep=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),numeric),("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore",max_categories=20))]),categorical)])
    x=prep.fit_transform(xraw); x=x.toarray() if hasattr(x,"toarray") else np.asarray(x)
    val=_real_split(ctx.root,"adult","R_final_test").sample(120,random_state=ctx.seed); yv=val.income.eq(">50K").astype(int).values; xv=prep.transform(feature_frame(val.drop(columns=["income"]))); xv=xv.toarray() if hasattr(xv,"toarray") else np.asarray(xv)
    return frame,x,y,source,xv,yv


def run_valuation(ctx:Context)->dict[str,Any]:
    frame,x,y,source,xv,yv=_valuation_matrix(ctx); results={}
    def knn_body(d:Path):
        toy_x=x[:8]; toy_y=y[:8]; toy_v=xv[:4]; toy_vy=yv[:4]
        fast=knn_shapley(toy_x,toy_y,toy_v,toy_vy,k=3); exact=exact_knn_shapley(toy_x,toy_y,toy_v,toy_vy,k=3); maxerr=float(np.max(np.abs(fast-exact)))
        values=knn_shapley(x,y,xv,yv,k=5); table=pd.DataFrame({"record_id":frame.mix_row_id.astype(str),"source_type":source,"value":values})
        metrics={"exact_verification_max_error":maxerr,"n":len(values),"real_mean":float(values[source=="real"].mean()),"synthetic_mean":float(values[source=="synthetic"].mean()),"negative_rate":float((values<0).mean()),"source_auroc":float(roc_auc_score(source=="synthetic",-values)),"_provenance":{"implementation":"paper_aligned_self_implementation","paper":"Jia et al. PVLDB 2019"}}
        return metrics,table
    results["V-KNN"]=ctx.run("V-KNN",knn_body,{"dataset":"adult","condition":"synthetic_replace","proportion":.25,"n":len(frame),"k":5})
    knn_values=pd.read_csv(results["V-KNN"]["directory"]/"predictions.csv"); knn_values.to_csv(ctx.out/"knn_shapley_values.csv",index=False)
    def oob_body(d:Path):
        values,coverage=data_oob(x,y,n_estimators=80,seed=ctx.seed); table=pd.DataFrame({"record_id":frame.mix_row_id.astype(str),"source_type":source,"value":values,"coverage":coverage})
        metrics={"n":len(values),"zero_coverage":int((coverage==0).sum()),"min_coverage":int(coverage.min()),"median_coverage":float(np.median(coverage)),"real_mean":float(np.nanmean(values[source=="real"])),"synthetic_mean":float(np.nanmean(values[source=="synthetic"])),"negative_rate":float(np.nanmean(values<0)),"spearman_with_knn":float(spearmanr(values,knn_values.value,nan_policy="omit").statistic),"_provenance":{"implementation":"paper_aligned_self_implementation","paper":"Kwon and Zou ICML 2023"}}
        return metrics,table
    results["V-OOB"]=ctx.run("V-OOB",oob_body,{"dataset":"adult","condition":"synthetic_replace","proportion":.25,"n":len(frame),"weak_learners":80})
    oob=pd.read_csv(results["V-OOB"]["directory"]/"predictions.csv"); oob.to_csv(ctx.out/"data_oob_values.csv",index=False)
    summaries=[]
    for aid,df in (("V-KNN",knn_values),("V-OOB",oob)):
        summaries.append({"algorithm_id":aid,"n":len(df),"real_mean":df.loc[df.source_type=="real","value"].mean(),"synthetic_mean":df.loc[df.source_type=="synthetic","value"].mean(),"negative_rate":np.mean(df.value<0)})
    pd.DataFrame(summaries).to_csv(ctx.out/"valuation_summary.csv",index=False)
    # Deletion evidence: source composition of the lowest-valued fractions.
    deletion=[]
    for aid,df in (("V-KNN",knn_values),("V-OOB",oob)):
        base_model=LogisticRegression(max_iter=500,random_state=ctx.seed).fit(x,y)
        baseline=float(roc_auc_score(yv,base_model.predict_proba(xv)[:,1]))
        for frac in (.05,.10,.20):
            selected=df.nsmallest(max(1,int(len(df)*frac)),"value")
            keep=np.ones(len(df),dtype=bool); keep[selected.index.to_numpy()]=False
            model=LogisticRegression(max_iter=500,random_state=ctx.seed).fit(x[keep],y[keep])
            utility=float(roc_auc_score(yv,model.predict_proba(xv)[:,1]))
            deletion.append({"algorithm_id":aid,"deleted_fraction":frac,"deleted_count":len(selected),
                "synthetic_fraction_deleted":np.mean(selected.source_type=="synthetic"),
                "baseline_real_test_auroc":baseline,"post_deletion_real_test_auroc":utility,
                "delta_real_test_auroc":utility-baseline})
    pd.DataFrame(deletion).to_csv(ctx.out/"valuation_deletion_curve.csv",index=False)
    return results


def code_audit(ctx:Context)->None:
    entries=[
      {"algorithm_id":"D1","paper":"Cross-table Synthetic Tabular Data Detection","paper_url":"https://aclanthology.org/2025.genaidetect-1.5/","searched_locations":["ACL Anthology","GitHub title/author search"],"official_code_url":None,"commit":None,"license":None,"retrieved_at":"2026-07-15","conclusion":"paper_aligned_reimplementation","evidence":"No official repository linked/found; three baselines are described in the paper."},
      {"algorithm_id":"D2","paper":"Synthetic Tabular Data Detection in the Wild","paper_url":"https://link.springer.com/chapter/10.1007/978-3-031-91398-3_7","searched_locations":["Springer","GitHub title/author search"],"official_code_url":None,"commit":None,"license":None,"retrieved_at":"2026-07-15","conclusion":"paper_aligned_reimplementation","evidence":"No verified official repository found."},
      {"algorithm_id":"D3","paper":"Robust Detection of Synthetic Tabular Data Under Schema Variability","paper_url":"https://ojs.aaai.org/index.php/AAAI/article/view/39422","searched_locations":["AAAI","arXiv","GitHub title/author search"],"official_code_url":None,"commit":None,"license":None,"retrieved_at":"2026-07-15","conclusion":"paper_aligned_reimplementation","evidence":"AAAI page/available version says code is to be released with an extended version; no verified official repo found."},
      {"algorithm_id":"QuaPy","paper":"QuaPy","paper_url":"https://github.com/HLT-ISTI/QuaPy","searched_locations":["official GitHub","PyPI"],"official_code_url":"https://github.com/HLT-ISTI/QuaPy","commit":"PyPI release 0.2.0","license":"BSD-3-Clause","retrieved_at":"2026-07-15","conclusion":"official_package_installed","evidence":"QuaPy 0.2.0 installed in isolated tabpollution environment."},
      {"algorithm_id":"V-KNN","paper":"Efficient Task-Specific Data Valuation for Nearest Neighbor Algorithms","paper_url":"https://www.vldb.org/pvldb/vol12/p1610-jia.pdf","searched_locations":["PVLDB","GitHub title/author search"],"official_code_url":None,"commit":None,"license":None,"retrieved_at":"2026-07-15","conclusion":"paper_aligned_reimplementation","evidence":"Recurrence implemented from primary paper and checked against exhaustive Shapley."},
      {"algorithm_id":"V-OOB","paper":"Data-OOB","paper_url":"https://proceedings.mlr.press/v202/kwon23e.html","searched_locations":["PMLR related material","GitHub title/author search"],"official_code_url":None,"commit":None,"license":None,"retrieved_at":"2026-07-15","conclusion":"paper_aligned_reimplementation","evidence":"Primary page exposes paper/OpenReview but no verified official code repository was found."},
    ]
    write_json(entries,ctx.reports/"code_availability_audit.json")
    lines=["# 算法代码可用性审计","","> 检索日期：2026-07-15。未找到不等于永久不存在，正式交稿前须复查。",""]
    for e in entries: lines += [f"## {e['algorithm_id']} — {e['paper']}","",f"- 结论：`{e['conclusion']}`",f"- 论文/仓库：{e['paper_url']}",f"- 官方代码：{e['official_code_url'] or '未核实到'}",f"- 版本/许可：{e['commit'] or '无'} / {e['license'] or '无'}",f"- 证据：{e['evidence']}",""]
    (ctx.reports/"code_availability_audit.md").write_text("\n".join(lines),encoding="utf-8")


def aggregate(ctx:Context,initial_hashes:dict[str,str],started:float)->None:
    final_hashes=_hashes(ctx.root); write_json({"before":initial_hashes,"after":final_hashes,"unchanged":initial_hashes==final_hashes},ctx.reports/"algorithm_runthrough_hash_regression.json")
    pd.DataFrame(ctx.run_index).to_csv(ctx.out/"run_index.csv",index=False)
    rows=[]
    paper={"D-LR":"Lopez-Paz & Oquab 2017 / Snoke et al. 2018","D-XGB":"Zein & Urvoy 2022","D-3G":"Kindji et al. 2025","D-FT":"Kindji et al. 2025","D-TT":"Kindji et al. 2025","D-DW":"Kindji et al. 2026","D-DWTA":"Kindji et al. 2026","V-KNN":"Jia et al. 2019","V-OOB":"Kwon & Zou 2023"}
    for aid,s in ctx.status.items():
        role=("经典" if aid in {"D-LR","D-XGB","Q-CC","Q-PCC","Q-ACC","Q-PACC","Q-EMQ","Q-HDy","Q-DyS","Q-MS","V-KNN","V-OOB"} else "直接基线" if aid in {"D-3G","D-FT","D-TT"} else "SOTA核心" if aid.startswith("D-DW") else "效用基线")
        rows.append({"algorithm_id":aid,"paper":paper.get(aid,"经典量化/效用方法"),"role":role,"implementation":"paper_aligned_self/local official package","data":"Adult GaussianCopula; cross-table methods also Adult/Credit/Abalone","protocol":"runthrough/smoke","status":s.get("status"),"key_result":json.dumps(s.get("metrics",{}),ensure_ascii=False),"difference_from_paper":"CPU tiny/single seed/subset; not paper performance reproduction","run_id":s.get("run_id",s.get("run_ids","")),"next_step":"formal 5-seed/full benchmark later"})
    status=pd.DataFrame(rows); status.to_csv(ctx.reports/"reproduction_status.csv",index=False)
    md_columns=["algorithm_id","paper","role","implementation","data","protocol","status","key_result","difference_from_paper","run_id","next_step"]
    def clean_cell(v:Any)->str: return str(v).replace("|","/").replace("\n"," ")
    md=["# 导师可查算法进展表","","> 本表全部为runthrough/smoke，不是正式论文性能复现。","",
        "|"+"|".join(md_columns)+"|","|"+"|".join(["---"]*len(md_columns))+"|"]
    for _,row in status.loc[:,md_columns].iterrows(): md.append("|"+"|".join(clean_cell(row[c]) for c in md_columns)+"|")
    (ctx.reports/"supervisor_algorithm_progress_table.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    detection=status[status.algorithm_id.str.startswith("D-")]
    det_rows=[]
    for r in detection.itertuples():
        m=ctx.status[r.algorithm_id].get("metrics",{}); det_rows.append({"algorithm_id":r.algorithm_id,"status":r.status,"run_id":r.run_id,**m})
    pd.DataFrame(det_rows).to_csv(ctx.out/"detection_summary.csv",index=False)
    scope="""# 论文设定与本地runthrough范围\n\n本轮所有结果均为`runthrough/smoke`，用于证明代码闭环，不能作为论文正式性能值。\n\n- D1/D2：本地仅三表mini fold，论文使用更完整跨表语料和协议。\n- D3：本地为论文对齐自实现、CPU tiny网络；未声称官方代码或论文AUC复现。\n- 量化：每比例3个test bags而非正式100 bags，统一复用D-XGB分数。\n- 效用：Adult+GaussianCopula+单seed子集，不是两表三生成器五seed。\n- 估值：Adult固定污染子集；KNN递推经过小N全子集核验，Data-OOB为最小bagging闭环。\n"""
    (ctx.out/"paper_vs_local_scope.md").write_text(scope,encoding="utf-8")
    passed=sum(v.get("status")=="runthrough_passed" for v in ctx.status.values()); failed=[k for k,v in ctx.status.items() if v.get("status")!="runthrough_passed"]
    actual_seconds=sum(float(v.get("seconds",0)) for v in ctx.status.values())
    report=f"""# 算法跑通交差版完成报告\n\n> 完成时间：{utc_now()}；成功run累计耗时：{actual_seconds:.1f}s；状态：{'通过' if not failed else '部分完成'}。\n\n## 结论\n\n必跑矩阵共{len(ctx.status)}项，已通过{passed}项，未通过{len(failed)}项：{failed or '无'}。所有结果均标记为runthrough/smoke，未进入formal汇总。\n\n## 实际范围\n\n- 经典检测：D-LR、D-XGB；直接论文基线：D-3G、D-FT、D-TT；D3核心：D-DW、D-DWTA。\n- 比例估计：CC、PCC、ACC、PACC、EMQ、HDy、DyS、Median Sweep，共用同一D-XGB分数和7档比例。\n- 估值：四条件×7比例的LR/XGBoost效用曲线、KNN-Shapley、Data-OOB。\n- 跨表mini数据：Adult、Credit、UCI Abalone；新增合成数据仅使用GaussianCopula。\n\n## 代码来源\n\nD1–D3、KNN-Shapley和Data-OOB均标记为论文对齐自实现；QuaPy 0.2.0为官方BSD-3-Clause包；XGBoost/PyTorch使用官方发行包。详见`code_availability_audit.md`。\n\n## 工程与保护\n\nC0–C3关键文件哈希是否不变：{initial_hashes==final_hashes}。本轮没有启动正式五种子、14表或完整两表三生成器矩阵。失败尝试保留在runs目录并由status.json标识，不进入成功汇总。\n\n## 必须保留的诊断警告\n\nAdult GaussianCopula的纯格式摘要诊断器结果见`algorithm_runthrough/format_artifact_diagnostic.json`。若AUROC高于0.65，说明序列化长度或缺失格式本身具有可检测性，因此D-XGB等极高P1结果不得直接解释为可泛化的语义检测能力。\n\n## 后续正式阶段仍需解决\n\nGPU环境、CTGAN/TVAE正式质量门、两表三生成器五seed、完整跨表表集、正式bag数、论文超参数及置信区间。当前产物足以证明算法代码已经跑通，并支撑回到文献综述和研究方案写作。\n"""
    (ctx.reports/"algorithm_runthrough_completion_report.md").write_text(report,encoding="utf-8")
    write_json({"run_type":"runthrough","created_at":utc_now(),"runs":ctx.run_index,"status_counts":{"passed":passed,"failed":len(failed)},"formal_inclusion":False},ctx.reports/"algorithm_runthrough_master_manifest.json")


def run_all(project_root:Path)->dict[str,Any]:
    root=project_root.resolve(); ctx=Context(root,root/"reports"); started=time.perf_counter(); initial=_hashes(root); code_audit(ctx)
    tables=prepare_three_tables(ctx); classical,models=run_classical(ctx); run_cross_table(ctx,tables); run_quantification(ctx,models["D-XGB"]); run_utility(ctx); run_valuation(ctx); aggregate(ctx,initial,started)
    return {"status":"complete","passed":sum(v.get("status")=="runthrough_passed" for v in ctx.status.values()),"total":len(ctx.status),"seconds":time.perf_counter()-started,"report":str(ctx.reports/"algorithm_runthrough_completion_report.md")}


def recover_and_aggregate(project_root:Path)->dict[str,Any]:
    """Recover completed runs after a report-only failure, without retraining."""
    root=project_root.resolve(); ctx=Context(root,root/"reports"); started=time.perf_counter(); initial=_hashes(root)
    complete=[]
    failed_attempts=[]
    for directory in root.glob("runs/runthrough-*"):
        status_path=directory/"status.json"
        if status_path.exists():
            status_data=json.loads(status_path.read_text(encoding="utf-8"))
            if status_data.get("status") in {"failed","blocked"}:
                failed_attempts.append({"run_id":directory.name,"status":status_data.get("status"),
                    "error":status_data.get("error",status_data.get("reason","")),"path":str(directory)})
        if not (directory/"COMPLETE").exists() or not (directory/"run_manifest.json").exists(): continue
        manifest=json.loads((directory/"run_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status")!="runthrough_passed": continue
        metrics=json.loads((directory/"metrics.json").read_text(encoding="utf-8")); timing=json.loads((directory/"timing.json").read_text(encoding="utf-8"))
        complete.append((directory.stat().st_mtime,directory,manifest,metrics,timing))
    for aid in ALGORITHM_IDS:
        matches=[x for x in complete if x[2].get("algorithm_id")==aid]
        if not matches: continue
        latest=max(matches,key=lambda x:x[0]); _,directory,manifest,metrics,timing=latest
        ctx.status[aid].update(status="runthrough_passed",run_id=manifest["run_id"],metrics={k:v for k,v in metrics.items() if k!="_provenance"},seconds=timing["total_seconds"])
        ctx.run_index.append({"algorithm_id":aid,"run_id":manifest["run_id"],"status":"runthrough_passed","seconds":timing["total_seconds"],"path":str(directory)})
        if aid=="D-3G": ctx.status[aid]["run_ids"]=[x[2]["run_id"] for x in sorted(matches,key=lambda x:x[0])[-3:]]
    pd.DataFrame(failed_attempts,columns=["run_id","status","error","path"]).to_csv(ctx.reports/"algorithm_runthrough_failed_attempts.csv",index=False)
    aggregate(ctx,initial,started)
    passed=sum(v.get("status")=="runthrough_passed" for v in ctx.status.values())
    return {"status":"complete","passed":passed,"total":len(ctx.status),"recovered_complete_runs":len(complete),
            "failed_attempts_preserved":len(failed_attempts),"report":str(ctx.reports/"algorithm_runthrough_completion_report.md")}
