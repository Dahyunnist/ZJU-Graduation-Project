"""Small runtime/package helpers shared by the frozen artificial gate and worker."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_package(package, expected_sha):
    package = Path(package).resolve()
    manifest_path = package / "package_manifest.json"
    if sha_file(manifest_path) != expected_sha:
        raise PermissionError("package commitment mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for rel, digest in manifest["files"].items():
        path = (package / rel).resolve()
        if not path.is_relative_to(package) or not path.is_file() or sha_file(path) != digest:
            raise PermissionError(f"package file mismatch: {rel}")
    actual = {p.relative_to(package).as_posix() for p in package.rglob("*")
              if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts}
    if actual != set(manifest["files"]) | {"package_manifest.json"}:
        raise PermissionError("unlisted package content")


def runtime_fingerprint(package):
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    spec = json.loads((Path(package) / "required_runtime.json").read_text(encoding="utf-8"))
    if not SpecifierSet(spec["python"]).contains(Version(platform.python_version())):
        raise RuntimeError("existing Python runtime does not match frozen requirements")
    distributions = {}
    pending = list(spec["required_distributions"])
    inspected = set()
    total_code_bytes = 0
    while pending:
        name = pending.pop()
        dist = importlib.metadata.distribution(name)
        canonical_name = dist.metadata["Name"].lower().replace("_", "-")
        if canonical_name in inspected:
            continue
        inspected.add(canonical_name)
        required = spec["required_distributions"].get(canonical_name)
        if required and not SpecifierSet(required).contains(Version(dist.version)):
            raise RuntimeError(f"existing {canonical_name} version does not match; installs and upgrades are disabled")
        files = {}
        for item in dist.files or ():
            path = Path(dist.locate_file(item))
            if path.is_file() and path.suffix.lower() in {".py", ".so", ".pyd", ".dll"}:
                total_code_bytes += path.stat().st_size
                if total_code_bytes > 8 * 1024**3:
                    raise RuntimeError("runtime dependency code snapshot exceeds 8 GiB")
                files[str(path.resolve())] = sha_file(path)
        distributions[canonical_name] = {"version": dist.version,
            "code_files_sha256": hashlib.sha256(canonical(files).encode()).hexdigest(),
            "code_file_count": len(files)}
        for dependency in dist.requires or ():
            requirement = Requirement(dependency)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                pending.append(requirement.name)
    runtime = {"python": sys.version, "executable_sha256": sha_file(sys.executable),
        "os": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "required_roots": spec["required_distributions"],
        "dependency_closure": distributions, "gpu": False}
    return runtime, hashlib.sha256(canonical(runtime).encode()).hexdigest()
