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
            'SMT': 'smt/results',
            'ReAct': 'react/results', 
            'GNN': 'gnn/results',
            'AnyTool': 'anytool/results',
            'MapReflect': 'map_reflect/results'
        }
    
    def get_result_file_path(self, method: str, dataset: str, model: str) -> Optional[Path]:
        """Get the path to a result file for method/dataset/model combination."""
        method_dir = self.method_dirs.get(method)
        if not method_dir:
            return None
        
        possible_names = [
            f"{dataset}_{model}.json",           # ITOps_granite.json
            f"{dataset.lower()}_{model}.json",   # itops_granite.json
            f"{dataset.upper()}_{model}.json",   # ITOPS_granite.json
        ]
        
        for name in possible_names:
            result_file = self.base_dir / method_dir / name
            if result_file.exists():
                return result_file
        
        return None
    
    def load_result_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load a single result file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
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
                        print(f"Missing result file: {method}/{dataset}/{model}")
                        continue
                    
                    result_data = self.load_result_file(result_file)
                    if not result_data:
                        continue
                    
                    # Process individual results from the file
                    method_results = self._process_result_data(result_data, dataset, method, model)
                    all_results.extend(method_results)
                    
                    print(f"Loaded {len(method_results)} results from {result_file}")
        
        return all_results
    
    def _process_result_data(self, result_data: Dict[str, Any], dataset: str, method: str, model: str) -> List[EvaluationResult]:
        """Convert raw result data to EvaluationResult objects."""
        results = []
        
        for result_entry in result_data.get('results', []):
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
            'overall': self._calculate_overall_stats(),
            'by_method': self._calculate_method_stats(),
            'by_dataset': self._calculate_dataset_stats(),
            'by_model': self._calculate_model_stats(),
            'combinations': self._calculate_combination_stats(),
            'by_dataset_model': self._calculate_dataset_model_stats()
        }
        
    def _calculate_dataset_model_stats(self) -> Dict[str, Dict[str, Any]]:
        """Calculate statistics for each dataset-model combination."""
        stats = {}
        
        for dataset in self.df['dataset'].unique():
            for model in self.df['model'].unique():
                combination_df = self.df[(self.df['dataset'] == dataset) & (self.df['model'] == model)]
                if len(combination_df) == 0:
                    continue
                    
                combination_key = f"{dataset}_{model}"
                successful = len(combination_df[combination_df['success'] == True])
                
                # Calculate method-specific stats for this dataset-model combination
                method_stats = {}
                for method in combination_df['method'].unique():
                    method_df = combination_df[combination_df['method'] == method]
                    method_successful = len(method_df[method_df['success'] == True])
                    
                    method_stats[method] = {
                        'count': len(method_df),
                        'success_rate': method_successful / len(method_df) if len(method_df) > 0 else 0.0,
                        'avg_accuracy': method_df['accuracy'].mean() if len(method_df) > 0 else 0.0,
                        'std_accuracy': method_df['accuracy'].std() if len(method_df) > 0 else 0.0
                    }
                
                stats[combination_key] = {
                    'dataset': dataset,
                    'model': model,
                    'total_count': len(combination_df),
                    'total_success_rate': successful / len(combination_df) if len(combination_df) > 0 else 0.0,
                    'overall_avg_accuracy': combination_df['accuracy'].mean() if len(combination_df) > 0 else 0.0,
                    'methods': method_stats
                }
        
        return stats
    
    def _calculate_overall_stats(self) -> Dict[str, Any]:
        """Calculate overall statistics."""
        total = len(self.df)
        successful = len(self.df[self.df['success'] == True])
        
        return {
            'total_evaluations': total,
            'successful_evaluations': successful,
            'success_rate': successful / total if total > 0 else 0.0,
            'avg_accuracy': self.df['accuracy'].mean() if total > 0 else 0.0,
            'std_accuracy': self.df['accuracy'].std() if total > 0 else 0.0,
            'avg_execution_time': self.df['execution_time'].mean() if total > 0 else 0.0
        }
    
    def _calculate_method_stats(self) -> Dict[str, Dict[str, Any]]:
        """Calculate statistics by method."""
        stats = {}
        
        for method in self.df['method'].unique():
            method_df = self.df[self.df['method'] == method]
            successful = len(method_df[method_df['success'] == True])
            
            stats[method] = {
                'count': len(method_df),
                'success_rate': successful / len(method_df) if len(method_df) > 0 else 0.0,
                'avg_accuracy': method_df['accuracy'].mean() if len(method_df) > 0 else 0.0,
                'std_accuracy': method_df['accuracy'].std() if len(method_df) > 0 else 0.0,
                'avg_execution_time': method_df['execution_time'].mean() if len(method_df) > 0 else 0.0
            }
        
        return stats
    
    def _calculate_dataset_stats(self) -> Dict[str, Dict[str, Any]]:
        """Calculate statistics by dataset."""
        stats = {}
        
        for dataset in self.df['dataset'].unique():
            dataset_df = self.df[self.df['dataset'] == dataset]
            successful = len(dataset_df[dataset_df['success'] == True])
            
            stats[dataset] = {
                'count': len(dataset_df),
                'success_rate': successful / len(dataset_df) if len(dataset_df) > 0 else 0.0,
                'avg_accuracy': dataset_df['accuracy'].mean() if len(dataset_df) > 0 else 0.0,
                'std_accuracy': dataset_df['accuracy'].std() if len(dataset_df) > 0 else 0.0
            }
        
        return stats
    
    def _calculate_model_stats(self) -> Dict[str, Dict[str, Any]]:
        """Calculate statistics by model."""
        stats = {}
        
        for model in self.df['model'].unique():
            model_df = self.df[self.df['model'] == model]
            successful = len(model_df[model_df['success'] == True])
            
            stats[model] = {
                'count': len(model_df),
                'success_rate': successful / len(model_df) if len(model_df) > 0 else 0.0,
                'avg_accuracy': model_df['accuracy'].mean() if len(model_df) > 0 else 0.0,
                'std_accuracy': model_df['accuracy'].std() if len(model_df) > 0 else 0.0
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
        """Save results and statistics to files."""
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        self.df.to_csv(output_dir / "detailed_results.csv", index=False)
        
        # Generate and save statistics
        stats = self.generate_statistics()
        if stats:
            with open(output_dir / "evaluation_statistics.json", 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            
            # Generate and save dataset-specific LaTeX tables
            self._generate_dataset_latex_tables(output_dir, stats)
            
            print(f"Results saved to {output_dir}")
            print(f"  - detailed_results.csv: {len(self.df)} individual results")
            print(f"  - evaluation_statistics.json: aggregated statistics")
            print(f"  - dataset_tables.tex: LaTeX tables for publication")
        else:
            print(f"Warning: No statistics generated")
            print(f"Results saved to {output_dir}")
            print(f"  - detailed_results.csv: {len(self.df)} individual results")
    
    def _generate_dataset_latex_tables(self, output_dir: Path, stats: Dict[str, Any]):
        """Generate dataset-specific LaTeX tables."""
        
        if not stats or not isinstance(stats, dict):
            print("Warning: Invalid statistics data, cannot generate LaTeX tables")
            return
        
        datasets = list(self.df['dataset'].unique())
        methods = ['SMT', 'ReAct', 'GNN', 'AnyTool', 'MapReflect']
        models = list(self.df['model'].unique())
        
        latex_content = []
        latex_content.append("% Generated Dataset-Specific LaTeX Tables")
        latex_content.append("% " + "=" * 60)
        latex_content.append("")
        
        # Generate separate table for each dataset
        for dataset in datasets:
            latex_content.append(f"% {dataset} Dataset Results")
            table_content = self._generate_single_dataset_table(dataset, methods, models, stats)
            if table_content:
                latex_content.append(table_content)
                latex_content.append("")
        
        # Generate combined latency table
        latex_content.append("% Combined Latency Results")
        latency_table = self._generate_combined_latency_table(datasets, methods, models, stats)
        if latency_table:
            latex_content.append(latency_table)
        
        # Save to file
        try:
            with open(output_dir / "dataset_tables.tex", 'w') as f:
                f.write("\n".join(latex_content))
            print(f"LaTeX tables generated successfully")
        except Exception as e:
            print(f"Error generating LaTeX tables: {e}")
    
    def _generate_single_dataset_table(self, dataset: str, methods: List[str], models: List[str], stats: Dict[str, Any]) -> str:
        """Generate table for a specific dataset."""
        
        if not stats or 'combinations' not in stats:
            print(f"Warning: No combinations data available for {dataset}")
            return ""
        
        combinations = stats['combinations']
        
        latex = []
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append(f"\\caption{{{dataset} Dataset Results by Model}}")
        latex.append(f"\\label{{tab:{dataset.lower()}_results}}")
        
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
            row = []
            if method == "MapReflect":
                row.append(f"\\textbf{{{method}}}")
            else:
                row.append(method)
            
            # Add accuracy for each model
            for model in models:
                combination_key = f"{method}_{dataset}_{model}"
                
                if combination_key in combinations:
                    accuracy = combinations[combination_key].get('avg_accuracy', 0.0)
                    accuracy_str = f"{accuracy:.3f}"
                    if method == "MapReflect":
                        row.append(f"\\textbf{{{accuracy_str}}}")
                    else:
                        row.append(accuracy_str)
                else:
                    row.append("--")
            
            latex.append(" & ".join(row) + " \\\\")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        return "\n".join(latex)
    
    def _generate_combined_latency_table(self, datasets: List[str], methods: List[str], models: List[str], stats: Dict[str, Any]) -> str:
        """Generate combined latency table for all datasets."""
        
        if not stats or 'combinations' not in stats:
            print("Warning: No combinations data available for latency table")
            return ""
        
        combinations = stats['combinations']
        
        latex = []
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append("\\caption{Execution Latency Results by Dataset and Model (seconds)}")
        latex.append("\\label{tab:latency_results}")
        
        col_spec = "l" + "r" * len(models)
        latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
        latex.append("\\toprule")
        
        for i, dataset in enumerate(datasets):
            # Dataset header
            latex.append(f"\\multicolumn{{{len(models) + 1}}}{{c}}{{\\textbf{{{dataset} Dataset}}}} \\\\")
            latex.append("\\midrule")
            
            # Column headers
            header = ["\\textbf{Method}"] + [f"\\textbf{{{model.title()}}}" for model in models]
            latex.append(" & ".join(header) + " \\\\")
            latex.append("\\midrule")
            
            # Method rows
            for method in methods:
                row = []
                if method == "MapReflect":
                    row.append(f"\\textbf{{{method}}}")
                else:
                    row.append(method)
                
                # Add latency for each model
                for model in models:
                    combination_key = f"{method}_{dataset}_{model}"
                    
                    if combination_key in combinations:
                        latency = combinations[combination_key].get('avg_execution_time', 0.0)
                        if latency > 0:
                            if method == "MapReflect":
                                row.append(f"\\textbf{{{latency:.1f}}}")
                            else:
                                row.append(f"{latency:.1f}")
                        else:
                            if method == "MapReflect":
                                row.append("\\textbf{TBA}")
                            else:
                                row.append("TBA")
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
    
    # Print summary
    stats = analyzer.generate_statistics()
    
    if stats is None:
        print("No stats from analyzer!")
        sys.exit(1)

    print("\nSummary Statistics:")
    print("=" * 20)
    
    overall = stats.get('overall', {})
    print(f"Total evaluations: {overall.get('total_evaluations', 0)}")
    print(f"Successful evaluations: {overall.get('successful_evaluations', 0)}")
    print(f"Overall success rate: {overall.get('success_rate', 0.0):.3f}")
    print(f"Average accuracy: {overall.get('avg_accuracy', 0.0):.3f}")
    
    print(f"\nMethod Performance:")
    for method, metrics in stats.get('by_method', {}).items():
        acc = metrics.get('avg_accuracy', 0.0)
        sr = metrics.get('success_rate', 0.0)
        count = metrics.get('count', 0)
        print(f"  {method}: {count} results, Accuracy={acc:.3f}, Success Rate={sr:.3f}")
    
    print(f"\nDataset Performance:")
    for dataset, metrics in stats.get('by_dataset', {}).items():
        acc = metrics.get('avg_accuracy', 0.0)
        count = metrics.get('count', 0)
        print(f"  {dataset}: {count} results, Accuracy={acc:.3f}")
    
    print(f"\nModel Performance:")
    for model, metrics in stats.get('by_model', {}).items():
        acc = metrics.get('avg_accuracy', 0.0)
        count = metrics.get('count', 0)
        print(f"  {model}: {count} results, Accuracy={acc:.3f}")
    
    # New detailed breakdown by dataset-model combinations
    print(f"\nDetailed Results by Dataset and Model:")
    print("=" * 50)
    
    by_dataset_model = stats.get('by_dataset_model', {})
    for combination_key, combination_data in by_dataset_model.items():
        dataset = combination_data['dataset']
        model = combination_data['model']
        print(f"\n{dataset} - {model}:")
        print(f"  Total: {combination_data['total_count']} results, Overall Accuracy: {combination_data['overall_avg_accuracy']:.3f}")
        
        methods = combination_data['methods']
        for method, method_stats in methods.items():
            count = method_stats['count']
            acc = method_stats['avg_accuracy']
            sr = method_stats['success_rate']
            print(f"    {method:12}: {count:3d} results, Accuracy={acc:.3f}, Success={sr:.1%}")


if __name__ == "__main__":
    main()
