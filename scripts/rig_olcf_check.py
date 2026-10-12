#!/usr/bin/env python
"""Check OLCF access through the AmSC RIG for both S3M enclaves.

Reads AMSC_TOKEN (your AmSC PAT) from the environment and, for each enclave:

1. confirms RIG advertises olcf-<enclave> as IRI v2, and
2. calls /api/v2/account/whoami through RIG, which only succeeds when the
   whole chain works: AmSC PAT -> RIG -> Credential Vault S3M token -> S3M.

Prints identities and reasons only, never tokens. Exit status 1 if any
enclave fails.

Usage:
    python scripts/rig_olcf_check.py [--enclave open|moderate|both]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "notebooks"))

from olcf_s3m import STAGING_RIG, RigAccessError, rig_whoami  # noqa: E402

STAGING_API = "https://api.staging.american-science-cloud.org/api/current"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--enclave", choices=["open", "moderate", "both"], default="both")
    parser.add_argument("--rig-url", default=STAGING_RIG)
    args = parser.parse_args(argv)

    token = os.environ.get("AMSC_TOKEN", "").strip()
    if not token:
        print("AMSC_TOKEN is not set (export your AmSC PAT first).")
        return 2

    from amsc_client import Client

    try:
        advertised = {
            f.name: f for f in Client(token=token, base_url=STAGING_API).rig_facilities(rig_url=args.rig_url)
        }
    except Exception as exc:  # report the type and message only
        print(f"RIG discovery failed: {type(exc).__name__}: {exc}")
        return 1

    ok = True
    for enclave in (["open", "moderate"] if args.enclave == "both" else [args.enclave]):
        name = f"olcf-{enclave}"
        meta = advertised.get(name)
        if meta is None:
            print(f"{name}: ❌ not advertised by RIG")
            ok = False
            continue
        print(f"{name}: advertised (tier {meta.tier}, api {meta.api_version}, probe {meta.probe_path})")
        try:
            identity = rig_whoami(token, enclave, rig_url=args.rig_url)
        except RigAccessError as exc:
            print(f"{name}: ❌ whoami refused — HTTP {exc.status}: {exc.detail}")
            ok = False
            continue
        fields = ", ".join(f"{k}={v}" for k, v in identity.items() if not str(k).startswith("_"))
        print(f"{name}: ✅ whoami through RIG — {fields}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
