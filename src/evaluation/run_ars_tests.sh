#!/usr/bin/env bash
# Run run_translator_tests_via_ars.py for assets with 5 <= GeneOrGeneProduct < 600,
# then merge all per-asset GMT files into a single combined GMT.
#
# Usage:
#   ./run_ars_tests.sh <tests-asset-dir> [output.gmt] [ars-env]
#
#   <tests-asset-dir>  Directory containing Asset_N.json files
#                      (clone of https://github.com/NCATSTranslator/Tests test_assets/)
#   [output.gmt]       Combined output GMT (default: data/combined_genes.gmt)
#   [ars-env]          ARS environment: prod|test|ci|dev (default: prod)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JSONL="${SCRIPT_DIR}/data/all_ars_test_results.jsonl"

TESTS_DIR="${1:?ERROR: missing required argument <tests-asset-dir>. Usage: $0 <tests-asset-dir> [output.gmt] [ars-env]}"
OUTPUT_GMT="${2:-${SCRIPT_DIR}/data/combined_genes.gmt}"
ARS_ENV="${3:-prod}"

TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

echo "=== Filtered ARS Test Runner ==="
echo "JSONL source  : $JSONL"
echo "Test assets   : $TESTS_DIR"
echo "Output GMT    : $OUTPUT_GMT"
echo "ARS env       : $ARS_ENV"
echo ""

# Write Python filter to a temp file to avoid quoting/heredoc edge cases
FILTER_PY="${TEMP_DIR}/filter.py"
cat > "$FILTER_PY" << 'PYEOF'
import json
import sys

jsonl_path = sys.argv[1]
min_genes = 5
max_genes = 600

with open(jsonl_path) as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        cats = record.get("ars_categories") or {}
        gene_count = cats.get("biolink:GeneOrGeneProduct", 0)
        if min_genes <= gene_count < max_genes:
            print(f"{record['test_asset']}\t{record['source_file']}")
PYEOF

MATCHING_TSV="${TEMP_DIR}/matching.tsv"
python3 "$FILTER_PY" "$JSONL" > "$MATCHING_TSV"

TOTAL=$(wc -l < "$MATCHING_TSV" | tr -d ' ')
echo "Assets matching 5 <= GeneOrGeneProduct < 600: ${TOTAL}"
echo ""

if [ "$TOTAL" -eq 0 ]; then
    echo "No matching assets found. Exiting."
    exit 0
fi

# Truncate/create the combined output file
> "$OUTPUT_GMT"

PROCESSED=0
SKIPPED=0

while IFS=$'\t' read -r asset_name source_file; do
    [ -z "$asset_name" ] && continue

    asset_path="${TESTS_DIR}/${source_file}"
    if [ ! -f "$asset_path" ]; then
        echo "[skip] ${asset_name}: file not found at ${asset_path}"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    temp_gmt="${TEMP_DIR}/${asset_name}.gmt"
    temp_jsonl="${TEMP_DIR}/${asset_name}.jsonl"

    echo "[${asset_name}] Submitting ${source_file} ..."
    python3 "${SCRIPT_DIR}/run_translator_tests_via_ars.py" \
        --tests-dir "$asset_path" \
        --ars-env  "$ARS_ENV" \
        --limit    0 \
        --output   "$temp_jsonl" \
        --gmt      "$temp_gmt" \
        -e

    if [ -f "$temp_gmt" ] && [ -s "$temp_gmt" ]; then
        GMT_LINES=$(wc -l < "$temp_gmt" | tr -d ' ')
        cat "$temp_gmt" >> "$OUTPUT_GMT"
        echo "[${asset_name}] Appended ${GMT_LINES} gene set(s)."
    else
        echo "[${asset_name}] No gene sets produced."
    fi

    PROCESSED=$((PROCESSED + 1))
done < "$MATCHING_TSV"

echo ""
echo "=== Done ==="
echo "Processed : ${PROCESSED}"
echo "Skipped   : ${SKIPPED} (asset file not found)"
FINAL_LINES=$(wc -l < "$OUTPUT_GMT" | tr -d ' ')
echo "Gene sets : ${FINAL_LINES} line(s) written to ${OUTPUT_GMT}"
