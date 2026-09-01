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
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

ARS_ENVS = {
    "prod": "https://ars-prod.transltr.io",
    "test": "https://ars.test.transltr.io",
    "ci": "https://ars.ci.transltr.io",
    "dev": "https://ars-dev.transltr.io",
}

BAYES_GENE_URL = "https://translator.broadinstitute.org/genetics_provider/bayes_gene/trapi_analysis"


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
    # ?trace=y tells ARS to include per-ARA child messages in the response
    url = f"{ars_base}/ars/api/messages/{pk}?trace=y"
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
                "pk": child.get("pk") or child.get("message"),
            }
        )
    return summary


def _extract_message(data: Dict[str, Any]) -> Dict[str, Any]:
    return (
        data.get("fields", {}).get("data", {}).get("message")
        or data.get("message")
        or {}
    )


def _genes_from_message(message: Dict[str, Any]) -> List[str]:
    """Extract unique gene names from TRAPI results via node_bindings and auxiliary graph edges."""
    kg_nodes = (message.get("knowledge_graph") or {}).get("nodes") or {}
    kg_edges = (message.get("knowledge_graph") or {}).get("edges") or {}
    aux_graphs = message.get("auxiliary_graphs") or {}

    candidate_curies: set = set()
    for result in (message.get("results") or []):
        # direct answer node bindings
        for bindings in (result.get("node_bindings") or {}).values():
            for b in (bindings or []):
                candidate_curies.add(b.get("id") if isinstance(b, dict) else b)
        # nodes reachable via auxiliary/support graphs
        for analysis in (result.get("analyses") or []):
            for graph_id in (analysis.get("support_graphs") or []):
                for edge_id in ((aux_graphs.get(graph_id) or {}).get("edges") or []):
                    edge = kg_edges.get(edge_id) or {}
                    candidate_curies.add(edge.get("subject"))
                    candidate_curies.add(edge.get("object"))

    genes: List[str] = []
    seen: set = set()
    for curie in candidate_curies:
        if not curie:
            continue
        node = kg_nodes.get(curie) or {}
        if "biolink:GeneOrGeneProduct" in (node.get("categories") or []):
            gene_name = node.get("name") or curie
            if gene_name not in seen:
                seen.add(gene_name)
                genes.append(gene_name)
    return genes


def fetch_kg_size(ars_base: str, child_pk: str, timeout: int = 30, save_path: Optional[Path] = None) -> Optional[tuple]:
    """Return (node_count, edge_count, category_counts, gene_names, message) from a child ARS message, or None on failure."""
    url = f"{ars_base}/ars/api/messages/{child_pk}"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        message = _extract_message(resp.json())
        if save_path is not None:
            save_path.write_text(json.dumps(resp.json(), indent=2))
            print(f"  submitting to bayes_gene: {BAYES_GENE_URL}")
            bayes_message = submit_bayes_analysis(resp.json())
            if bayes_message is not None:
                bayes_path = Path(save_path).parent / f"{save_path.stem}_bayes_gene.json"
                bayes_path.write_text(json.dumps(bayes_message, indent=2))
                factors = (bayes_message.get("pigean-factor") or {}).get("data") or []
                labels = [f.get("label") or f.get("factor", "?") for f in factors]
                print(f"  bayes_gene -> {len(factors)} factor(s), saved to {bayes_path.name}")
                for label in labels:
                    print(f"    * {label}")
        kg = message.get("knowledge_graph") or {}
        nodes = kg.get("nodes") or {}
        edges = kg.get("edges") or {}
        cats: Counter = Counter()
        for node in nodes.values():
            for cat in (node.get("categories") or ["unknown"]):
                cats[cat] += 1
        return len(nodes), len(edges), dict(cats.most_common()), _genes_from_message(message), message
    except Exception:
        return None


def fetch_kg_genes(ars_base: str, child_pk: str, timeout: int = 30) -> Optional[List[str]]:
    """Return gene names from TRAPI results via node_bindings and auxiliary graphs, or None on failure."""
    url = f"{ars_base}/ars/api/messages/{child_pk}"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return _genes_from_message(_extract_message(resp.json()))
    except Exception:
        return None


def submit_bayes_analysis(message: Dict[str, Any], timeout: int = 600) -> Optional[Dict[str, Any]]:
    """POST a TRAPI message to the Bayes Gene analysis endpoint and return the response message."""
    try:
        resp = requests.post(BAYES_GENE_URL, json=message, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"  [warn] bayes_gene analysis failed: {e}")
        return None


def run(args: argparse.Namespace) -> None:
    ars_base = ARS_ENVS[args.ars_env]
    verbose = args.verbose
    tests_path = Path(args.tests_dir)
    if not tests_path.exists():
        sys.exit(f"Path not found: {tests_path}")

    assets = load_test_assets(tests_path)
    print(f"Loaded {len(assets)} test asset(s) from {tests_path}")

    if args.limit:
        assets = assets[: args.limit]

    results = []
    out_path = Path(args.output)
    out_fh = out_path.open("w")
    gmt_fh = open(args.gmt, "w") if args.gmt else None
    for i, asset in enumerate(assets, 1):
        name = asset.get("id") or asset.get("name") or f"asset_{i}"
        print(f"\n[{i}/{len(assets)}] {name} (from {asset.get('_source_file')})")
        try:
            trapi_payload = build_trapi_query(asset)
            if trapi_payload is None:
                print("  [skip] could not build a TRAPI query (missing id/predicate) -- check schema mapping")
                continue

            if verbose:
                print(f"  POST {ars_base}/ars/api/submit")
            try:
                pk = submit_to_ars(ars_base, trapi_payload)
            except requests.RequestException as e:
                print(f"  [error] submit failed: {e}")
                continue

            if not pk:
                continue
            arax_url = f"https://arax.transltr.io/?source=ARS&id={pk}"
            if verbose:
                print(f"  submitted -> pk={pk}")
                print(f"  polling {ars_base}/ars/api/messages/{pk}?trace=y")
            print(f"  ARAX UI: {arax_url}")

            ars_result = poll_ars(
                ars_base, pk, max_wait_s=args.max_wait, poll_every_s=args.poll_interval, debug=args.debug
            )
            child_summary = summarize_children(ars_result)
            if not child_summary:
                top_keys = list(ars_result.keys())
                fields_keys = list(ars_result.get("fields", {}).keys())
                print(f"  [warn] no per-ARA children parsed -- top-level keys: {top_keys}, fields keys: {fields_keys}")
                print(f"  [warn] rerun with --debug to print the full raw response")
            ars_kg: Optional[tuple] = None
            for c in child_summary:
                kg_info = ""
                cat_lines = []
                is_ars = c.get("agent") == "ars-ars-agent"
                is_done = (c.get("status") or "").lower() in ("done", "complete", "completed")
                if is_ars and c.get("pk") and is_done:
                    save_path = Path(args.output).parent / f"{name}_ars.json" if args.eaggl_analysis else None
                    size = fetch_kg_size(ars_base, c["pk"], save_path=save_path)
                    if size is not None:
                        ars_kg = size
                        kg_info = f" [nodes={size[0]} edges={size[1]}]"
                        if verbose:
                            cat_lines = [f"      {cat}: {cnt}" for cat, cnt in size[2].items()]
                        else:
                            cat_lines = [f"      {cat}: {cnt}" for cat, cnt in size[2].items() if cat == "biolink:GeneOrGeneProduct"]
                        if gmt_fh and len(size[3]) >= 5:
                            gmt_fh.write("\t".join([name, f"pk:{pk}"] + size[3]) + "\n")
                            gmt_fh.flush()
                elif gmt_fh and c.get("pk") and is_done:
                    genes = fetch_kg_genes(ars_base, c["pk"])
                    if genes and len(genes) >= 0:
                        agent = c.get("agent", "unknown")
                        gmt_fh.write("\t".join([f"{name}_{agent}", agent] + genes) + "\n")
                        gmt_fh.flush()
                if verbose:
                    print(f"    - {c['agent']:<20} status={c['status']} code={c['code']} pk={c.get('pk')}{kg_info}")
                elif is_ars and kg_info:
                    print(f"    - {'ars-ars-agent':<20} status={c['status']} code={c['code']}{kg_info}")
                for line in cat_lines:
                    print(line)

            record = {
                "test_asset": name,
                "source_file": asset.get("_source_file"),
                "pk": pk,
                "ars_env": args.ars_env,
                "ars_nodes": ars_kg[0] if ars_kg else None,
                "ars_edges": ars_kg[1] if ars_kg else None,
                "ars_categories": ars_kg[2] if ars_kg else None,
                "ars_genes": ars_kg[3] if ars_kg else None,
                "children": child_summary,
            }
            results.append(record)
            out_fh.write(json.dumps(record) + "\n")
            out_fh.flush()

        except Exception as e:
            print(f"  [error] unexpected failure for {name}: {e}")

        if args.sleep:
            time.sleep(args.sleep)

    out_fh.close()
    if gmt_fh:
        gmt_fh.close()
        print(f"Wrote gene sets to {args.gmt}")
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
    p.add_argument("--output", default="data/ars_test_results.jsonl", help="Where to write the summary (JSON Lines)")
    p.add_argument("--debug", action="store_true", help="Print raw ARS JSON responses while polling")
    p.add_argument("-v", "--verbose", action="store_true", help="Print per-ARA children, all categories, and request URLs")
    p.add_argument("-e", "--eaggl-analysis", action="store_true", help="Save the ars-ars-agent TRAPI response and run EAGGL/Bayes Gene analysis for each asset")
    p.add_argument("-gmt", "--gmt", metavar="FILE", default=None, help="Write gene sets to this GMT file (aggregate >=5 genes, per-ARA >=3 genes)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
