"""Fail-closed validation and loading for candidate role-assignment artifacts."""
from __future__ import annotations
import csv,hashlib,json,re
from pathlib import Path
class CandidateRoleError(PermissionError):
    """Candidate role metadata or artifact failed its pre-fit gate."""
def _sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()
def _inside_package(package,relative):
    if not isinstance(relative,str) or not relative:raise CandidateRoleError("candidate role artifact path is required")
    p=Path(relative)
    if p.is_absolute():raise CandidateRoleError("candidate role artifact path must be package-relative")
    q=package/p
    if q.is_symlink():raise CandidateRoleError("candidate role artifact must not be a symlink")
    q=q.resolve(strict=True)
    if not q.is_relative_to(package):raise CandidateRoleError("candidate role artifact is outside the verified package")
    if not q.is_file():raise CandidateRoleError("candidate role artifact must be a regular file")
    return q
def validate_candidate_role_manifest(package_root,config):
    package=Path(package_root).resolve(strict=True)
    rel=config.get("role_candidate_manifest_file");expected=config.get("role_candidate_manifest_sha256")
    if not isinstance(expected,str) or not re.fullmatch(r"[0-9a-f]{64}",expected):raise CandidateRoleError("candidate role manifest SHA-256 is required")
    path=_inside_package(package,rel)
    if _sha(path)!=expected:raise CandidateRoleError("candidate role manifest SHA-256 mismatch")
    manifest=json.loads(path.read_text(encoding="utf-8"));entries=manifest.get("entries")
    if not isinstance(entries,list):raise CandidateRoleError("candidate role manifest entries are malformed")
    by_id={row.get("dataset_id"):row for row in entries};ids=[row.get("dataset_id") for row in config.get("datasets",[])]
    if len(by_id)!=len(entries) or set(by_id)!=set(ids):raise CandidateRoleError("candidate role manifest dataset set mismatch")
    verified=[];blockers=[]
    for dataset in ids:
        row=by_id[dataset];mode=row.get("role_mode");status=row.get("status")
        if mode=="artifact":
            artifact=_inside_package(package,row.get("role_assignment_file"))
            if _sha(artifact)!=row.get("role_assignment_sha256"):raise CandidateRoleError(f"role assignment SHA-256 mismatch for {dataset}")
            if row.get("rows") is None or row.get("rows")<1:raise CandidateRoleError(f"role assignment row count missing for {dataset}")
            with artifact.open("r",encoding="utf-8",newline="") as f:
                reader=csv.DictReader(f)
                if reader.fieldnames!=["record_id","role"]:raise CandidateRoleError(f"role assignment schema mismatch for {dataset}")
                records=list(reader)
            if len(records)!=row["rows"]:raise CandidateRoleError(f"role assignment row count mismatch for {dataset}")
            if [x["record_id"] for x in records]!=[f"public:{dataset}:{i}" for i in range(len(records))]:raise CandidateRoleError(f"role assignment source-order IDs mismatch for {dataset}")
            counts={}
            for x in records:
                role=x["role"]
                if role not in {"train","tune","evaluation","final_test","purged"}:raise CandidateRoleError(f"unknown role label for {dataset}: {role!r}")
                counts[role]=counts.get(role,0)+1
            if counts!=row.get("role_counts"):raise CandidateRoleError(f"role assignment counts mismatch for {dataset}")
            if not {"train","tune","evaluation","final_test"}<=counts.keys():raise CandidateRoleError(f"required role is empty for {dataset}")
            if row.get("candidate_source_file"):
                source=_inside_package(package,row["candidate_source_file"])
                if _sha(source)!=row.get("candidate_source_file_sha256"):raise CandidateRoleError(f"source-bound candidate file SHA-256 mismatch for {dataset}")
                with source.open("r",encoding="utf-8",newline="") as stream:
                    source_reader=csv.reader(stream)
                    source_fields=next(source_reader,[])
                    source_rows=sum(1 for _ in source_reader)
                expected_fields=row.get("source_column_order")
                actual_fields=[name for name in source_fields if name not in {"role","record_id","source_order"}]
                if actual_fields!=expected_fields:raise CandidateRoleError(f"candidate source field order mismatch for {dataset}")
                if source_rows!=row.get("candidate_source_rows"):raise CandidateRoleError(f"candidate source row count mismatch for {dataset}")
        elif mode=="runtime_reconstruction":
            digest=row.get("role_assignment_sha256")
            if not isinstance(digest,str) or not re.fullmatch(r"[0-9a-f]{64}",digest):
                blockers.append(dataset)
        elif mode=="unavailable":blockers.append(dataset)
        else:raise CandidateRoleError(f"unsupported role artifact mode for {dataset}: {mode!r}")
        if status=="blocked" and dataset not in blockers:blockers.append(dataset)
        verified.append(dataset)
    return {"status":"verified_with_blockers" if blockers else "verified","verified_datasets":verified,"blocked_datasets":blockers,"execution_enabled":manifest.get("execution_enabled") is True,"manifest_sha256":expected}
def require_candidate_role_manifest_ready(package_root,config):
    result=validate_candidate_role_manifest(package_root,config)
    if result["blocked_datasets"]:raise CandidateRoleError("source and role identities required; candidate role manifest has unresolved datasets: "+", ".join(result["blocked_datasets"]))
    if result["execution_enabled"] is not True or config.get("execution",{}).get("enabled") is not True:raise CandidateRoleError("candidate role artifacts are not frozen or authorized for execution")
    return result
def load_candidate_role_assignment(package_root,spec,row_count):
    package=Path(package_root).resolve(strict=True);dataset=spec["dataset_id"]
    path=_inside_package(package,spec.get("candidate_role_assignment_file"));expected=spec.get("candidate_role_assignment_sha256")
    if not isinstance(expected,str) or _sha(path)!=expected:raise CandidateRoleError(f"role assignment SHA-256 mismatch for {dataset}")
    with path.open("r",encoding="utf-8",newline="") as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=["record_id","role"]:raise CandidateRoleError(f"role assignment schema mismatch for {dataset}")
        rows=list(reader)
    if len(rows)!=int(row_count):raise CandidateRoleError(f"role assignment row count mismatch for {dataset}")
    if [x["record_id"] for x in rows]!=[f"public:{dataset}:{i}" for i in range(len(rows))]:raise CandidateRoleError(f"role assignment source-order IDs mismatch for {dataset}")
    import pandas as pd
    roles=pd.DataFrame(rows,columns=["record_id","role"])
    if set(roles.role)-{"train","tune","evaluation","final_test","purged"}:raise CandidateRoleError(f"unknown role in candidate assignment for {dataset}")
    if not {"train","tune","evaluation","final_test"}<=set(roles.role):raise CandidateRoleError(f"required role is empty for {dataset}")
    return roles


def verify_release_code_manifest(package_root, manifest_name="package_manifest.json"):
    """Verify code-only release inventory; external role artifacts are never package files."""
    package=Path(package_root).resolve(strict=True)
    manifest_path=_inside_package(package,manifest_name)
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_type")!="code_only_external_roles":
        raise CandidateRoleError("code package manifest type is invalid")
    files=manifest.get("files")
    if not isinstance(files,dict) or not files:
        raise CandidateRoleError("code package manifest file inventory is empty")
    for relative,digest in files.items():
        try:
            path=_inside_package(package,relative)
        except (CandidateRoleError,FileNotFoundError) as exc:
            raise CandidateRoleError(f"code package inventory mismatch; missing or unsafe file: {relative}") from exc
        if not isinstance(digest,str) or not re.fullmatch(r"[0-9a-f]{64}",digest) or _sha(path)!=digest:
            raise CandidateRoleError(f"code package file digest mismatch: {relative}")
    actual={p.relative_to(package).as_posix() for p in package.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts}
    expected=set(files)|{manifest_name}
    if actual!=expected:
        missing=sorted(expected-actual);extra=sorted(actual-expected)
        raise CandidateRoleError(f"code package inventory mismatch; missing={missing}; extra={extra}")
    return {"status":"verified","file_count":len(files)}

def require_external_role_dependencies(package_root,config,*,private_manifest_path=None):
    """Verify server-private role files before source access; missing files fail closed."""
    package=Path(package_root).resolve(strict=True)
    code_manifest=config.get("code_package_manifest_file","package_manifest.json")
    verify_release_code_manifest(package,code_manifest)
    public_path=_inside_package(package,config.get("external_role_dependencies_file"))
    public_digest=config.get("external_role_dependencies_sha256")
    if not isinstance(public_digest,str) or _sha(public_path)!=public_digest:
        raise CandidateRoleError("external role dependency declaration digest mismatch")
    public=json.loads(public_path.read_text(encoding="utf-8"))
    if public.get("manifest_id")!="broad_utility_v2_external_role_dependency_declaration":
        raise CandidateRoleError("external role dependency declaration is malformed")
    env_name=public.get("private_manifest_env")
    if not isinstance(env_name,str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*",env_name):
        raise CandidateRoleError("private role manifest environment key is malformed")
    if private_manifest_path is None:
        import os
        private_manifest_path=os.environ.get(env_name)
    if not private_manifest_path:
        raise CandidateRoleError("server-private role manifest is required")
    private_path=Path(private_manifest_path)
    if private_path.is_symlink() or not private_path.is_file():
        raise CandidateRoleError("server-private role manifest is missing or unsafe")
    private_path=private_path.resolve(strict=True);private_root=private_path.parent
    try: private_path.relative_to(package)
    except ValueError: pass
    else: raise CandidateRoleError("private role manifest must stay outside code package")
    private=json.loads(private_path.read_text(encoding="utf-8"))
    if private.get("manifest_id")!="broad_utility_v2_private_role_dependency_manifest":
        raise CandidateRoleError("server-private role manifest is malformed")
    required=public.get("required_datasets");blocked=public.get("blocked_datasets")
    if not isinstance(required,list) or len(set(required))!=len(required) or not isinstance(blocked,list):
        raise CandidateRoleError("external role dataset inventory is malformed")
    entries=private.get("entries");by_id={x.get("dataset_id"):x for x in entries or []}
    expected=set(required)|set(blocked)
    if not isinstance(entries,list) or len(by_id)!=len(entries) or set(by_id)!=expected:
        raise CandidateRoleError("server-private role inventory does not match public declaration")
    for dataset in required:
        row=by_id[dataset]
        if row.get("status")!="available": raise CandidateRoleError(f"private role artifact unavailable: {dataset}")
        relative=row.get("artifact_path");p=Path(relative) if isinstance(relative,str) else Path("/")
        if p.is_absolute() or not relative: raise CandidateRoleError(f"private role artifact path invalid: {dataset}")
        candidate=private_root/p
        if candidate.is_symlink() or not candidate.is_file(): raise CandidateRoleError(f"private role artifact missing: {dataset}")
        candidate=candidate.resolve(strict=True)
        if not candidate.is_relative_to(private_root): raise CandidateRoleError(f"private role artifact escapes its root: {dataset}")
        digest=row.get("sha256")
        if not isinstance(digest,str) or not re.fullmatch(r"[0-9a-f]{64}",digest) or _sha(candidate)!=digest:
            raise CandidateRoleError(f"private role artifact digest mismatch: {dataset}")
        with candidate.open("r",encoding="utf-8",newline="") as stream:
            reader=csv.DictReader(stream)
            if reader.fieldnames!=["record_id","role"]: raise CandidateRoleError(f"private role schema mismatch: {dataset}")
            records=list(reader)
        if len(records)!=row.get("rows") or [r["record_id"] for r in records]!=[f"public:{dataset}:{i}" for i in range(len(records))]:
            raise CandidateRoleError(f"private role row identity mismatch: {dataset}")
        counts={}
        for record in records:
            role=record["role"]
            if role not in {"train","tune","evaluation","final_test","purged"}: raise CandidateRoleError(f"private role label invalid: {dataset}")
            counts[role]=counts.get(role,0)+1
        if counts!=row.get("role_counts") or not {"train","tune","evaluation","final_test"}<=counts.keys():
            raise CandidateRoleError(f"private role counts mismatch: {dataset}")
    for dataset in blocked:
        row=by_id[dataset]
        if row.get("status")!="blocked" or any(k in row for k in ("artifact_path","sha256","rows","role_counts")):
            raise CandidateRoleError(f"blocked private dependency must not expose an artifact locator or digest: {dataset}")
    if blocked:
        raise CandidateRoleError("unresolved private role dependencies: "+", ".join(blocked))
    if public.get("execution_enabled") is not True or config.get("execution",{}).get("enabled") is not True:
        raise CandidateRoleError("external role artifacts verified but execution is not authorized")
    return {"status":"verified","required_datasets":required}
