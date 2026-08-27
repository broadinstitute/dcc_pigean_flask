#!/usr/bin/env python3
"""
run_translator_tests_via_ars.py

Loads NCATS Translator test assets (from a local clone of
https://github.com/NCATSTranslator/Tests) and submits them as TRAPI
one-hop queries to the ARS (Autonomous Relay System), then polls for
and summarizes the results.

SETUP
-----
    pip install requests
    git clone https://github.com/NCATSTranslator/Tests.git

USAGE
-----
    python run_translator_tests_via_ars.py --tests-dir ./Tests/test_assets --limit 5

    # point at a single file instead of a whole directory
    python run_translator_tests_via_ars.py --tests-dir ./Tests/test_assets/some_file.json

IMPORTANT
---------
- ARS has several deployments, pick one with --ars-env:
    prod -> https://ars-prod.transltr.io   (the live production system)
    test -> https://ars.test.transltr.io   (default here -- safer to hammer)
    ci   -> https://ars.ci.transltr.io
    dev  -> https://ars-dev.transltr.io
- Please don't point --limit at "all of them" against prod. Start small
  (--limit 5), and use --sleep / --poll-interval to be polite to shared
  infrastructure that other Translator teams also depend on.
- Test Asset JSON schemas have varied over the life of this project.
  This script tries several historically-used key names in
  build_trapi_query() below, but you should open one real file from
  your local Tests/test_assets/ clone and confirm the field names
  match, adjusting the `_first(...)` calls if not.
- Likewise, the exact shape of the ARS /ars/api/messages/{pk} response
  has drifted across ARS versions. summarize_children() tries a couple
  of common shapes; if it comes back empty, run with --debug once to
  print the raw JSON and adjust it to match what you actually get.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

ARS_ENVS = {
    "prod": "https://ars-prod.transltr.io",
    "test": "https://ars.test.transltr.io",
    "ci": "https://ars.ci.transltr.io",
    "dev": "https://ars-dev.transltr.io",
}


def load_test_assets(path: Path) -> List[Dict[str, Any]]:
    """Load one Test Asset JSON file, or every *.json file in a directory."""
    files = [path] if path.is_file() else sorted(path.glob("*.json"))
    assets = []
    for f in files:
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            print(f"  [skip] {f.name}: invalid JSON ({e})", file=sys.stderr)
            continue
        # A file might hold a single test asset dict, or a list of them.
        items = data if isinstance(data, list) else [data]
        for item in items:
            if isinstance(item, dict):
                item["_source_file"] = f.name
                assets.append(item)
    return assets


def _first(d: Dict[str, Any], *keys: str, default=None):
    """Return the first present, non-None value among several candidate key names."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def build_trapi_query(asset: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert a Translator Test Asset into a one-hop TRAPI query graph.

    Checks several historically-used key names for the same concept.
    ADJUST THIS to match your actual test asset schema -- inspect a
    real file first (e.g. `head -50 Tests/test_assets/*.json`).
    """
    subject_id = _first(asset, "subject_id", "subject", "input_id", "input_curie")
    subject_category = _first(
        asset, "subject_category", "input_category", default="biolink:NamedThing"
    )
    predicate = _first(asset, "predicate_id", "predicate", default="biolink:related_to")
    object_id = _first(asset, "object_id", "object", "output_id", "output_curie")
    object_category = _first(
        asset, "object_category", "output_category", default="biolink:NamedThing"
    )
    # "by_subject" = subject id known, solve for object (most common shape).
    # "by_object"  = object id known, solve for subject.
    direction = _first(asset, "direction", "test_direction", default="by_subject")

    if not predicate:
        return None
    if not predicate.startswith("biolink:"):
        predicate = f"biolink:{predicate}"

    qg_subject: Dict[str, Any] = {"categories": [subject_category]}
    qg_object: Dict[str, Any] = {"categories": [object_category]}

    if direction == "by_object":
        if not object_id:
            return None
        qg_object["ids"] = [object_id]
    else:
        if not subject_id:
            return None
        qg_subject["ids"] = [subject_id]

    return {
        "message": {
            "query_graph": {
                "nodes": {"n0": qg_subject, "n1": qg_object},
                "edges": {
                    "e0": {"subject": "n0", "object": "n1", "predicates": [predicate]}
                },
            }
        },
        "submitter": "translator-test-script",
    }


def submit_to_ars(ars_base: str, trapi_payload: Dict[str, Any], timeout: int = 30) -> Optional[str]:
    """POST a TRAPI request to the ARS /submit endpoint. Returns the parent pk (UUID)."""
    url = f"{ars_base}/ars/api/submit"
    resp = requests.post(url, json=trapi_payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    pk = data.get("pk") or data.get("message") or data.get("job_id")
    if not pk:
        print("  [warn] no pk found in submit response:", json.dumps(data)[:300])
    return pk


def poll_ars(
    ars_base: str,
    pk: str,
    max_wait_s: int = 300,
    poll_every_s: int = 10,
    debug: bool = False,
) -> Dict[str, Any]:
    """Poll /ars/api/messages/{pk} until the parent query is Done/Error or timeout."""
    url = f"{ars_base}/ars/api/messages/{pk}"
    waited = 0
    last: Dict[str, Any] = {}
    while waited <= max_wait_s:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        last = resp.json()
        if debug:
            print(json.dumps(last, indent=2)[:2000])
        status = last.get("status") or last.get("fields", {}).get("status")
        if status and status.lower() in ("done", "error", "complete", "completed"):
            return last
        time.sleep(poll_every_s)
        waited += poll_every_s
    print(f"  [warn] pk={pk} did not finish within {max_wait_s}s (last status={last.get('status')})")
    return last


def summarize_children(ars_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Pull a simple per-ARA summary (name, status, http code) out of the ARS response."""
    children = ars_result.get("children") or ars_result.get("fields", {}).get("children") or []
    summary = []
    for child in children:
        actor = child.get("actor") if isinstance(child.get("actor"), dict) else {}
        name = actor.get("agent") or actor.get("name") or child.get("name") or "unknown"
        summary.append(
            {
                "agent": name,
                "status": child.get("status"),
                "code": child.get("code"),
            }
        )
    return summary


def run(args: argparse.Namespace) -> None:
    ars_base = ARS_ENVS[args.ars_env]
    tests_path = Path(args.tests_dir)
    if not tests_path.exists():
        sys.exit(f"Path not found: {tests_path}")

    assets = load_test_assets(tests_path)
    print(f"Loaded {len(assets)} test asset(s) from {tests_path}")

    if args.limit:
        assets = assets[: args.limit]

    results = []
    for i, asset in enumerate(assets, 1):
        name = asset.get("id") or asset.get("name") or f"asset_{i}"
        print(f"\n[{i}/{len(assets)}] {name} (from {asset.get('_source_file')})")

        trapi_payload = build_trapi_query(asset)
        if trapi_payload is None:
            print("  [skip] could not build a TRAPI query (missing id/predicate) -- check schema mapping")
            continue

        try:
            pk = submit_to_ars(ars_base, trapi_payload)
        except requests.RequestException as e:
            print(f"  [error] submit failed: {e}")
            continue

        if not pk:
            continue
        print(f"  submitted -> pk={pk}")
        print(f"  view in ARAX UI: https://arax.transltr.io/?source=ARS&id={pk}")

        ars_result = poll_ars(
            ars_base, pk, max_wait_s=args.max_wait, poll_every_s=args.poll_interval, debug=args.debug
        )
        child_summary = summarize_children(ars_result)
        if not child_summary:
            print("  [warn] no per-ARA children parsed -- rerun with --debug to inspect raw response shape")
        for c in child_summary:
            print(f"    - {c['agent']:<20} status={c['status']} code={c['code']}")

        results.append(
            {
                "test_asset": name,
                "source_file": asset.get("_source_file"),
                "pk": pk,
                "ars_env": args.ars_env,
                "children": child_summary,
            }
        )

        if args.sleep:
            time.sleep(args.sleep)

    out_path = Path(args.output)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results)} result(s) to {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--tests-dir", required=True, help="Path to a Test Asset JSON file, or a directory of them"
    )
    p.add_argument(
        "--ars-env", choices=list(ARS_ENVS.keys()), default="test",
        help="Which ARS deployment to hit (default: test, NOT prod)",
    )
    p.add_argument("--limit", type=int, default=5, help="Max number of tests to run (default: 5)")
    p.add_argument("--max-wait", type=int, default=300, help="Max seconds to wait for each query")
    p.add_argument("--poll-interval", type=int, default=10, help="Seconds between status polls")
    p.add_argument("--sleep", type=float, default=1.0, help="Seconds to pause between submissions")
    p.add_argument("--output", default="ars_test_results.json", help="Where to write the summary JSON")
    p.add_argument("--debug", action="store_true", help="Print raw ARS JSON responses while polling")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
