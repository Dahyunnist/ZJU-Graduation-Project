# Broad Utility V2: code-only release

This release contains source code, run configuration, and source metadata only. It contains no role assignment CSVs.

The public external-role declaration lists required and blocked datasets but contains no private paths, role counts, or artifact digests. Before source access, the loader validates this code package and validates each required private role artifact against the server-private manifest named by `TABPOLLUTION_PRIVATE_ROLE_MANIFEST`. Missing, modified, malformed, or blocked dependencies cause refusal.

The private manifest and role artifacts stay outside this release. The blocked dataset entries contain no artifact path or digest. The sealed source has no locator or digest here.

Execution remains disabled. The package supports integrity and fail-closed checks only. No model fitting or formal exploration ran.
