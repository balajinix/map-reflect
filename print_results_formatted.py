#!/usr/bin/env python3
"""Print results in formatted box-drawing tables."""

import subprocess
import json

def get_results():
    """Run calc_accuracy.py and parse output."""
    result = subprocess.run(
        ['python3', 'eval/calc_accuracy.py', '--output', 'json'],
        capture_output=True, text=True
    )
    # Filter out warning lines - find where JSON starts
    output = result.stdout
    json_start = output.find('{')
    if json_start > 0:
        output = output[json_start:]
    return json.loads(output)

def print_box_table(title, subtitle, headers, rows, col_widths):
    """Print a table with box-drawing characters."""
    total_width = sum(col_widths) + len(col_widths) + 1

    # Top border
    print("┌" + "─" * (total_width - 2) + "┐")

    # Title
    print("│" + title.center(total_width - 2) + "│")
    print("│" + subtitle.center(total_width - 2) + "│")

    # Header separator
    print("├" + "┬".join("─" * w for w in col_widths) + "┤")

    # Headers
    header_str = "│"
    for i, h in enumerate(headers):
        header_str += h.center(col_widths[i]) + "│"
    print(header_str)

    # Header-data separator
    print("├" + "┼".join("─" * w for w in col_widths) + "┤")

    # Data rows
    prev_method = None
    for row in rows:
        # Add separator between methods
        if prev_method is not None and row[0] != prev_method and row[0] != "":
            print("├" + "┼".join("─" * w for w in col_widths) + "┤")
        prev_method = row[0] if row[0] else prev_method

        row_str = "│"
        for i, cell in enumerate(row):
            row_str += str(cell).center(col_widths[i]) + "│"
        print(row_str)

    # Bottom border
    print("└" + "┴".join("─" * w for w in col_widths) + "┘")

def main():
    results = get_results()

    datasets = ['ITOpsGraph', 'NL2GQL', 'OMOP_CMS', 'OMOP_MIMIC', 'OMOP_Synthea']

    # Define method order and models
    method_order = [
        ('SMT', 'Granite'),
        ('SMT', 'Llama'),
        ('GNN', '(no LLM)'),
        ('ReAct', 'Granite'),
        ('ReAct', 'Llama'),
        ('AnyTool', 'Granite'),
        ('AnyTool', 'Llama'),
        ('Map-Reflect', 'Granite'),
        ('Map-Reflect', 'Llama'),
        ('MatchMaker', 'Granite'),
        ('MatchMaker', 'Llama'),
        ('ReMatch', 'Granite'),
        ('ReMatch', 'Llama'),
        ('UE-Reflect', 'Granite'),
        ('UE-Reflect', 'Llama'),
        ('UE-Reflect-V2', 'Granite'),
        ('UE-Reflect-V2', 'Llama'),
        ('UE-Reflect-V3', 'Granite'),
        ('UE-Reflect-V3', 'Llama'),
    ]

    headers = ['Method', 'Model'] + datasets
    col_widths = [14, 12] + [12] * len(datasets)

    print()
    print("=" * 100)
    print("COMPREHENSIVE RESULTS: ALL METHODS, MODELS, AND DATASETS")
    print("=" * 100)
    print()

    # TABLE 1: API-LEVEL ACCURACY
    rows = []
    for method, model in method_order:
        row = [method, model]
        has_data = False
        for ds in datasets:
            key = f"{method}|{model}|{ds}"
            if key in results:
                api_acc = results[key].get('api_accuracy', 0) * 100
                row.append(f"{api_acc:.1f}%")
                has_data = True
            else:
                row.append("--")
        if has_data:
            rows.append(row)

    print_box_table(
        "TABLE 1: API-LEVEL (PRODUCT) ACCURACY",
        "Correct when predicted mapping contains fields from expected APIs",
        headers, rows, col_widths
    )

    print()

    # TABLE 2: FIELD-LEVEL ACCURACY
    rows = []
    for method, model in method_order:
        row = [method, model]
        has_data = False
        for ds in datasets:
            key = f"{method}|{model}|{ds}"
            if key in results:
                r = results[key]
                if r.get('api_correct_with_gt', 0) > 0:
                    field_acc = r.get('field_accuracy', 0) * 100
                    row.append(f"{field_acc:.1f}%")
                    has_data = True
                else:
                    row.append("N/A")
            else:
                row.append("--")
        if has_data:
            rows.append(row)

    print_box_table(
        "TABLE 2: FIELD-LEVEL ACCURACY (Given Correct API)",
        "Field mapping accuracy ONLY where API selection was correct",
        headers, rows, col_widths
    )

    print()

    # TABLE 3: SAMPLE COUNTS
    rows = []
    for method, model in method_order:
        row = [method, model]
        has_data = False
        for ds in datasets:
            key = f"{method}|{model}|{ds}"
            if key in results:
                r = results[key]
                api_correct = r.get('api_correct', 0)
                total = r.get('samples', 0)
                row.append(f"{api_correct}/{total}")
                has_data = True
            else:
                row.append("--")
        if has_data:
            rows.append(row)

    print_box_table(
        "TABLE 3: SAMPLE COUNTS (API Correct / Total)",
        "",
        headers, rows, col_widths
    )

if __name__ == "__main__":
    main()
