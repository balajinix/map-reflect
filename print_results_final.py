#!/usr/bin/env python3
"""Print results in the exact format requested."""

import subprocess
import json

def get_results():
    """Run calc_accuracy.py and parse output."""
    result = subprocess.run(
        ['python3', 'eval/calc_accuracy.py', '--output', 'json'],
        capture_output=True, text=True
    )
    return json.loads(result.stdout)

def print_table(title, subtitle, headers, rows, col_widths):
    """Print a table with box-drawing characters."""
    total_width = sum(col_widths) + len(col_widths) + 1

    # Top border
    print("┌" + "─" * (total_width - 2) + "┐")

    # Title
    print("│" + title.center(total_width - 2) + "│")
    if subtitle:
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
    # Baselines: SMT, GNN, ReAct, AnyTool, MatchMaker, ReMatch
    # Our methods: Map-Reflect, UE-Reflect
    method_order = [
        ('SMT', 'Granite'),
        ('SMT', 'Llama'),
        ('GNN', '(no LLM)'),
        ('ReAct', 'Granite'),
        ('ReAct', 'Llama'),
        ('AnyTool', 'Granite'),
        ('AnyTool', 'Llama'),
        ('MatchMaker', 'Granite'),
        ('MatchMaker', 'Llama'),
        ('ReMatch', 'Granite'),
        ('ReMatch', 'Llama'),
        ('Map-Reflect', 'Granite'),
        ('Map-Reflect', 'Llama'),
        ('UE-Reflect', 'Granite'),
        ('UE-Reflect', 'Llama'),
    ]

    headers = ['Method', 'Model', 'ITOpsGraph', 'NL2GQL', 'OMOP_CMS', 'OMOP_MIMIC', 'OMOP_Synthea']
    col_widths = [14, 12, 8, 8, 10, 12, 13]

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

    print_table(
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

    print_table(
        "TABLE 2: FIELD-LEVEL ACCURACY (Given Correct API)",
        "Field mapping accuracy ONLY where API selection was correct",
        headers, rows, col_widths
    )

    print()

    # TABLE 3: SAMPLE COUNTS
    headers3 = ['Method', 'Model', 'ITOpsGraph', 'NL2GQL', 'OMOP_CMS', 'OMOP_MIMIC', 'OMOP_Synth']
    col_widths3 = [14, 12, 10, 10, 10, 12, 11]

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

    print_table(
        "TABLE 3: SAMPLE COUNTS (API Correct / Total)",
        "",
        headers3, rows, col_widths3
    )

if __name__ == "__main__":
    main()
