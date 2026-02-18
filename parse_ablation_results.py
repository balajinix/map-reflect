#!/usr/bin/env python3
"""
Parse ablation experiment results and generate LaTeX table for the paper.
"""

import json
import os
from pathlib import Path

ABLATION_DIR = Path("results/ablation")

EXPERIMENTS = [
    ("base", "Base", "---", "---", "---"),
    ("neighborhood", "+Neighborhood", "\\checkmark", "---", "---"),
    ("reflection", "+Reflection", "---", "\\checkmark", "---"),
    ("uq", "+UQ", "---", "---", "\\checkmark"),
    ("neighborhood_reflection", "+Nbr+Refl", "\\checkmark", "\\checkmark", "---"),
    ("neighborhood_uq", "+Nbr+UQ", "\\checkmark", "---", "\\checkmark"),
    ("reflection_uq", "+Refl+UQ", "---", "\\checkmark", "\\checkmark"),
    ("full", "Full", "\\checkmark", "\\checkmark", "\\checkmark"),
]

def parse_results(exp_dir):
    """Parse metrics from experiment results."""
    result_files = list(exp_dir.glob("*.json"))
    if not result_files:
        return None
    
    for rf in result_files:
        try:
            with open(rf) as f:
                data = json.load(f)
            metrics = data.get("metrics", data)
            return {
                "api_accuracy": metrics.get("api_accuracy", 0) * 100,
                "field_accuracy": metrics.get("accuracy", metrics.get("field_accuracy", 0)) * 100,
                "total_samples": metrics.get("total_samples", 0),
                "total_fields": metrics.get("total_fields", 0),
            }
        except Exception as e:
            print(f"Error parsing {rf}: {e}")
    return None

def generate_latex_table():
    """Generate LaTeX table for ablation results."""
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Ablation Study: Impact of Individual Components on ITOpsGraph (Llama)}")
    print("\\label{tab:ablation_components}")
    print("\\begin{tabular}{l|ccc|rr}")
    print("\\toprule")
    print("\\textbf{Configuration} & \\textbf{Nbr} & \\textbf{Refl} & \\textbf{UQ} & \\textbf{API \\%} & \\textbf{Field \\%} \\\\")
    print("\\midrule")
    
    for exp_name, label, nbr, refl, uq in EXPERIMENTS:
        exp_dir = ABLATION_DIR / exp_name
        if exp_dir.exists():
            metrics = parse_results(exp_dir)
            if metrics:
                api_acc = f"{metrics['api_accuracy']:.1f}"
                field_acc = f"{metrics['field_accuracy']:.1f}"
            else:
                api_acc = "--"
                field_acc = "--"
        else:
            api_acc = "--"
            field_acc = "--"
        
        print(f"{label} & {nbr} & {refl} & {uq} & {api_acc} & {field_acc} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

def print_summary():
    """Print summary of results."""
    print("\n" + "="*60)
    print("ABLATION RESULTS SUMMARY")
    print("="*60)
    
    results = []
    for exp_name, label, _, _, _ in EXPERIMENTS:
        exp_dir = ABLATION_DIR / exp_name
        if exp_dir.exists():
            metrics = parse_results(exp_dir)
            if metrics:
                results.append((label, metrics))
                print(f"{label:20s}: API={metrics['api_accuracy']:.1f}%, Field={metrics['field_accuracy']:.1f}%")
            else:
                print(f"{label:20s}: Results not found")
        else:
            print(f"{label:20s}: Not run yet")
    
    if results:
        print("\n" + "-"*60)
        base = next((m for l, m in results if l == "Base"), None)
        full = next((m for l, m in results if l == "Full"), None)
        if base and full:
            print(f"Improvement from Base to Full:")
            print(f"  API:   {base['api_accuracy']:.1f}% -> {full['api_accuracy']:.1f}% (+{full['api_accuracy']-base['api_accuracy']:.1f}pp)")
            print(f"  Field: {base['field_accuracy']:.1f}% -> {full['field_accuracy']:.1f}% (+{full['field_accuracy']-base['field_accuracy']:.1f}pp)")

if __name__ == "__main__":
    print_summary()
    print("\n" + "="*60)
    print("LATEX TABLE")
    print("="*60 + "\n")
    generate_latex_table()
