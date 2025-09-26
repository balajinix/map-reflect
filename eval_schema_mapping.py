#!/usr/bin/env python3
"""
Schema Mapping Evaluation - Result Aggregator
Collects existing results from method directories and generates evaluation scores.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import pandas as pd


@dataclass
class EvaluationResult:
    """Results for a single schema mapping evaluation."""
    dataset: str
    method: str
    model: str
    sample_id: str
    predicted_mapping: Dict[str, Any]
    ground_truth_mapping: Dict[str, Any]
    accuracy: float
    execution_time: float
    success: bool = True
    error_message: Optional[str] = None


class ResultCollector:
    """Collect results from method-specific directories."""
    
    def __init__(self, base_dir: Path = Path(".")):
        self.base_dir = Path(base_dir)
        self.method_dirs = {
            'smt': 'smt/results',
            'react': 'react/results', 
            'gnn': 'gnn/results',
            'anytool': 'anytool/results',
            'map_reflect': 'map_reflect/results'
        }
    
    def get_result_file_path(self, method: str, dataset: str, model: str) -> Optional[Path]:
        """Get the path to a result file for method/dataset/model combination."""
        method_dir = self.method_dirs.get(method.lower())
        if not method_dir:
            return None
        
        possible_names = [
            f"{dataset}_{model}.jsonl",
            f"{dataset.lower()}_{model}.jsonl",
            f"{dataset.upper()}_{model}.jsonl",
            f"{dataset}_{model}.json",           # ITOps_granite.json
            f"{dataset.lower()}_{model}.json",   # itops_granite.json
            f"{dataset.upper()}_{model}.json",   # ITOPS_granite.json
        ]
        
        for name in possible_names:
            result_file = self.base_dir / method_dir / name
            if result_file.exists():
                return result_file
        
        return None
    
    def load_result_file(self, file_path: Path) -> Optional[List[Dict[str, Any]]]:
        """Load a result file - supports both JSONL and JSON formats."""
        try:
            if file_path.suffix == '.jsonl':
                # Load JSONL file (one JSON object per line)
                results = []
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:  # Skip empty lines
                            results.append(json.loads(line))
                return results
            else:
                # Load JSON file (legacy format)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # If it's a dict with 'results' key, extract the results list
                    if isinstance(data, dict) and 'results' in data:
                        return data['results']
                    # If it's already a list, return as-is
                    elif isinstance(data, list):
                        return data
                    # If it's a single object, wrap in list
                    else:
                        return [data]
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def collect_all_results(self, datasets: List[str], methods: List[str], models: List[str]) -> List[EvaluationResult]:
        """Collect all available results for the specified combinations."""
        all_results = []
        
        for dataset in datasets:
            for method in methods:
                for model in models:
                    result_file = self.get_result_file_path(method, dataset, model)
                    
                    if not result_file:
                        method_dir = self.method_dirs.get(method.lower(), f"{method.lower()}/results/")
                        print(f"Missing result file: {method_dir}/{dataset}_{model}.jsonl")
                        continue
                    
                    result_data = self.load_result_file(result_file)
                    if not result_data:
                        continue
                    
                    method_results = self._process_result_data(result_data, dataset, method, model)
                    all_results.extend(method_results)
                    
                    print(f"Loaded {len(method_results)} results from {result_file}")
        
        return all_results
    
    def _process_result_data(self, result_data: List[Dict[str, Any]], dataset: str, method: str, model: str) -> List[EvaluationResult]:
        """Convert raw result data from JSONL or JSON to EvaluationResult objects."""
        results = []
        
        for result_entry in result_data:
            try:
                # Calculate accuracy if not provided
                accuracy = result_entry.get('accuracy')
                if accuracy is None:
                    predicted = result_entry.get('predicted_mapping', {})
                    ground_truth = result_entry.get('ground_truth_mapping', {})
                    accuracy = self._calculate_accuracy(ground_truth, predicted)
                
                result = EvaluationResult(
                    dataset=dataset,
                    method=method,
                    model=model,
                    sample_id=result_entry.get('sample_id', ''),
                    predicted_mapping=result_entry.get('predicted_mapping', {}),
                    ground_truth_mapping=result_entry.get('ground_truth_mapping', {}),
                    accuracy=accuracy,
                    execution_time=result_entry.get('execution_time', 0.0),
                    success=result_entry.get('success', True),
                    error_message=result_entry.get('error_message')
                )
                results.append(result)
                
            except Exception as e:
                print(f"Error processing result entry in {dataset}_{method}_{model}: {e}")
                continue
        
        return results
    
    def _calculate_accuracy(self, ground_truth: Dict[str, Any], predicted: Dict[str, Any]) -> float:
        """Calculate accuracy for schema mapping with flexible field matching."""
        if not ground_truth:
            return 0.0
        
        if not predicted:
            return 0.0

        # Exact match check first
        gt_pairs = set(ground_truth.items())
        pred_pairs = set(predicted.items())
        exact_matches = len(gt_pairs.intersection(pred_pairs))

        if exact_matches > 0:
            return exact_matches / len(gt_pairs)

        # Fuzzy matching for partial credit
        correct_mappings = 0
        for gt_source, gt_target in ground_truth.items():
            for pred_source, pred_target in predicted.items():
                if (gt_source.lower() in pred_source.lower() or pred_source.lower() in gt_source.lower()) and \
                   (gt_target.lower() in pred_target.lower() or pred_target.lower() in gt_target.lower()):
                    correct_mappings += 1
                    break

        return correct_mappings / len(ground_truth)


class EvaluationAnalyzer:
    """Generate analysis and statistics from collected results."""
    
    def __init__(self, results: List[EvaluationResult]):
        self.results = results
        self.df = pd.DataFrame([asdict(result) for result in results])
    
    def generate_statistics(self) -> Dict[str, Any]:
        """Generate comprehensive statistics."""
        if self.df.empty:
            return {}
        
        stats = {
            'combinations': self._calculate_combination_stats()
        }
        
        return stats
    
    def _calculate_combination_stats(self) -> Dict[str, Dict[str, Any]]:
        """Calculate statistics for each method/dataset/model combination."""
        stats = {}
        
        for _, row in self.df.groupby(['method', 'dataset', 'model']).agg({
            'accuracy': ['mean', 'std', 'count'],
            'execution_time': 'mean',
            'success': 'sum'
        }).iterrows():
            method, dataset, model = _
            combination_key = f"{method}_{dataset}_{model}"
            
            count = row[('accuracy', 'count')]
            success_count = row[('success', 'sum')]
            
            stats[combination_key] = {
                'method': method,
                'dataset': dataset,
                'model': model,
                'count': count,
                'success_rate': success_count / count if count > 0 else 0.0,
                'avg_accuracy': row[('accuracy', 'mean')] if not pd.isna(row[('accuracy', 'mean')]) else 0.0,
                'std_accuracy': row[('accuracy', 'std')] if not pd.isna(row[('accuracy', 'std')]) else 0.0,
                'avg_execution_time': row[('execution_time', 'mean')] if not pd.isna(row[('execution_time', 'mean')]) else 0.0
            }
        
        return stats
    
    def save_results(self, output_dir: Path):
        """Save results in LaTeX format only."""
        output_dir.mkdir(exist_ok=True)
        
        # Generate and save statistics
        stats = self.generate_statistics()
        if stats:
            # Print detailed results first
            self._print_detailed_results(stats)
            
            # Generate and print LaTeX tables
            latex_content = self._generate_latex_tables(stats)
            print("\n" + "="*80)
            print("LATEX TABLES")
            print("="*80)
            print(latex_content)
            print("="*80)
            
            # Save LaTeX to file
            with open(output_dir / "results_tables.tex", 'w') as f:
                f.write(latex_content)
            
            print(f"\nLaTeX tables saved to: {output_dir}/results_tables.tex")
        else:
            print("Warning: No statistics generated")
    
    def _print_detailed_results(self, stats: Dict[str, Any]):
        """Print detailed results by dataset and model."""
        print("\nDetailed Results by Dataset and Model:")
        print("=" * 50)
        
        combinations = stats.get('combinations', {})
        if not combinations:
            print("No combination data available")
            return
        
        # Group by dataset and model
        dataset_model_combos = {}
        for combo_key, combo_data in combinations.items():
            dataset = combo_data['dataset']
            model = combo_data['model']
            method = combo_data['method']
            
            dm_key = f"{dataset}_{model}"
            if dm_key not in dataset_model_combos:
                dataset_model_combos[dm_key] = {
                    'dataset': dataset,
                    'model': model,
                    'methods': {},
                    'total_count': 0,
                    'total_accuracy_sum': 0.0
                }
            
            dataset_model_combos[dm_key]['methods'][method] = combo_data
            dataset_model_combos[dm_key]['total_count'] += combo_data['count']
            dataset_model_combos[dm_key]['total_accuracy_sum'] += combo_data['avg_accuracy'] * combo_data['count']
        
        # Print results grouped by dataset-model
        for dm_key in sorted(dataset_model_combos.keys()):
            combo = dataset_model_combos[dm_key]
            dataset = combo['dataset']
            model = combo['model']
            total_count = combo['total_count']
            overall_accuracy = combo['total_accuracy_sum'] / total_count if total_count > 0 else 0.0
            
            print(f"\n{dataset} - {model}:")
            print(f"  Total: {total_count} results, Overall Accuracy: {overall_accuracy:.3f}")
            
            # Print method results
            for method in ['smt', 'react', 'gnn', 'anytool', 'map_reflect']:
                if method in combo['methods']:
                    method_data = combo['methods'][method]
                    count = method_data['count']
                    accuracy = method_data['avg_accuracy']
                    success_rate = method_data['success_rate']
                    print(f"    {method:12}: {int(count):3d} results, Accuracy={accuracy:.3f}, Success={success_rate:.1%}")
                else:
                    print(f"    {method:12}: No results")
    
    def _generate_latex_tables(self, stats: Dict[str, Any]) -> str:
        """Generate comprehensive LaTeX tables for all results."""
        
        if not stats or 'combinations' not in stats:
            return "% No data available for LaTeX tables"
        
        combinations = stats['combinations']
        datasets = sorted(self.df['dataset'].unique())
        methods = ['smt', 'react', 'gnn', 'anytool', 'map_reflect']
        models = sorted(self.df['model'].unique())
        
        latex_content = []
        latex_content.append("% Schema Mapping Evaluation Results")
        latex_content.append("% Generated automatically")
        latex_content.append("")
        
        # Generate accuracy table for each dataset
        for dataset in datasets:
            latex_content.append(f"% {dataset.upper()} Dataset - Accuracy Results")
            table_content = self._generate_accuracy_table(dataset, methods, models, combinations)
            if table_content:
                latex_content.append(table_content)
                latex_content.append("")
        
        # Generate execution time table
        latex_content.append("% Execution Time Results (seconds)")
        time_table = self._generate_time_table(datasets, methods, models, combinations)
        if time_table:
            latex_content.append(time_table)
        
        return "\n".join(latex_content)
    
    def _generate_accuracy_table(self, dataset: str, methods: List[str], models: List[str], combinations: Dict[str, Any]) -> str:
        """Generate accuracy table for a specific dataset."""
        
        latex = []
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append(f"\\caption{{Accuracy Results for {dataset.upper()} Dataset}}")
        latex.append(f"\\label{{tab:{dataset.lower()}_accuracy}}")
        
        # Column specification: method name + one column per model
        col_spec = "l" + "r" * len(models)
        latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
        latex.append("\\toprule")
        
        # Header row
        header = ["\\textbf{Method}"] + [f"\\textbf{{{model.title()}}}" for model in models]
        latex.append(" & ".join(header) + " \\\\")
        latex.append("\\midrule")
        
        # Data rows
        for method in methods:
            row = [method.upper()]
            
            # Add accuracy for each model
            for model in models:
                combination_key = f"{method}_{dataset}_{model}"
                
                if combination_key in combinations:
                    accuracy = combinations[combination_key].get('avg_accuracy', 0.0)
                    count = combinations[combination_key].get('count', 0)
                    if count > 0:
                        row.append(f"{accuracy:.3f}")
                    else:
                        row.append("--")
                else:
                    row.append("--")
            
            latex.append(" & ".join(row) + " \\\\")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        return "\n".join(latex)
    
    def _generate_time_table(self, datasets: List[str], methods: List[str], models: List[str], combinations: Dict[str, Any]) -> str:
        """Generate execution time table for all datasets."""
        
        latex = []
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append("\\caption{Execution Time Results by Dataset and Model (seconds)}")
        latex.append("\\label{tab:execution_times}")
        
        col_spec = "l" + "r" * len(models)
        latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
        latex.append("\\toprule")
        
        for i, dataset in enumerate(datasets):
            # Dataset header
            latex.append(f"\\multicolumn{{{len(models) + 1}}}{{c}}{{\\textbf{{{dataset.upper()} Dataset}}}} \\\\")
            latex.append("\\midrule")
            
            # Column headers
            header = ["\\textbf{Method}"] + [f"\\textbf{{{model.title()}}}" for model in models]
            latex.append(" & ".join(header) + " \\\\")
            latex.append("\\midrule")
            
            # Method rows
            for method in methods:
                row = [method.upper()]
                
                # Add latency for each model
                for model in models:
                    combination_key = f"{method}_{dataset}_{model}"
                    
                    if combination_key in combinations:
                        time_val = combinations[combination_key].get('avg_execution_time', 0.0)
                        count = combinations[combination_key].get('count', 0)
                        if count > 0 and time_val > 0:
                            row.append(f"{time_val:.1f}")
                        else:
                            row.append("--")
                    else:
                        row.append("--")
                
                latex.append(" & ".join(row) + " \\\\")
            
            # Add separator between datasets (except for last one)
            if i < len(datasets) - 1:
                latex.append("\\midrule")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        return "\n".join(latex)


def main():
    """Main entry point for simplified evaluation."""
    parser = argparse.ArgumentParser(description="Schema Mapping Evaluation - Result Aggregator")
    parser.add_argument("--datasets", nargs='+', default=['ITOps', 'NL2GQL'],
                       help="Datasets to collect results for")
    parser.add_argument("--methods", nargs='+', default=['SMT', 'ReAct', 'GNN', 'AnyTool', 'MapReflect'],
                       help="Methods to collect results for")
    parser.add_argument("--models", nargs='+', default=['granite', 'llama', 'mistral-large'],
                       help="Models to collect results for")
    parser.add_argument("--output-dir", default="./results",
                       help="Output directory for aggregated results")
    parser.add_argument("--base-dir", default=".",
                       help="Base directory containing method subdirectories")
    
    args = parser.parse_args()
    
    print("Schema Mapping Evaluation")
    print("=" * 40)
    print(f"Datasets: {args.datasets}")
    print(f"Methods: {args.methods}")
    print(f"Models: {args.models}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Collect results
    collector = ResultCollector(Path(args.base_dir))
    results = collector.collect_all_results(args.datasets, args.methods, args.models)
    
    if not results:
        print("No results found!")
        sys.exit(1)
    
    print(f"Collected {len(results)} evaluation results")
    
    # Generate analysis
    analyzer = EvaluationAnalyzer(results)
    analyzer.save_results(Path(args.output_dir))


if __name__ == "__main__":
    main()
