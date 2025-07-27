#!/usr/bin/env python3
"""
Map-Reflect: GraphQL Schema Mapping Evaluation Framework
Comprehensive evaluation script for AAAI paper submission
"""

import asyncio
import json
import time
import os
import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Results for a single schema mapping evaluation."""
    dataset: str
    method: str
    model: str
    sample_id: str
    source_schema: str
    target_schema: str
    ground_truth_mapping: Dict[str, Any]
    predicted_mapping: Dict[str, Any]
    accuracy: float
    execution_time: float
    error_message: Optional[str] = None
    success: bool = True


class DatasetLoader:
    """Load and process GraphQL and REST API datasets."""
    
    def __init__(self):
        self.datasets = {
            'ITOps': 'https://github.ibm.com/solis-dal/solis-dal-dataset/blob/main/itops_dataset.jsonl',
            'NL2GQL': 'https://github.ibm.com/GraphQL-Query-Gen/NL2GQL',
            'GraphQLRestBench': 'https://github.com/GraphQL-Gen/GraphQLRestBench'
        }
    
    def get_expected_samples_count(self, dataset_name: str) -> int:
        """Get expected number of samples for a dataset."""
        expected_counts = {
            'ITOps': 100,
            'NL2GQL': 50,
            'GraphQLRestBench': 75
        }
        return expected_counts.get(dataset_name, 0)


class ResultFileLoader:
    """Load results from method-specific JSON files."""
    
    def __init__(self, base_dir: Path = Path(".")):
        self.base_dir = Path(base_dir)
        self.method_dirs = {
            'SMT': 'smt',
            'AnyTool': 'anytool', 
            'ReAct': 'react',
            'GNN': 'gnn',
            'Map-Reflect': 'map_reflect'
        }
    
    def load_method_results(self, method: str, dataset: str, model: str) -> Optional[Dict[str, Any]]:
        """
        Load results for a specific method/dataset/model combination.
        Expected file path: <method_dir>/results/<dataset>_<model>.json
        """
        method_dir = self.method_dirs.get(method)
        if not method_dir:
            logger.error(f"Unknown method directory mapping for: {method}")
            return None
        
        results_file = self.base_dir / method_dir / "results" / f"{dataset}_{model}.json"
        
        if not results_file.exists():
            logger.warning(f"Missing result file: {results_file}")
            return None
        
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            logger.info(f"✅ Loaded results from {results_file}")
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {results_file}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error reading {results_file}: {e}")
            return None
    
    def check_file_coverage(self, methods: List[str], datasets: List[str], models: List[str]) -> Dict[str, Any]:
        """Check which result files exist and which are missing."""
        total_expected = len(methods) * len(datasets) * len(models)
        existing_files = []
        missing_files = []
        
        for method in methods:
            for dataset in datasets:
                for model in models:
                    method_dir = self.method_dirs.get(method, method.lower())
                    results_file = self.base_dir / method_dir / "results" / f"{dataset}_{model}.json"
                    
                    if results_file.exists():
                        existing_files.append(str(results_file))
                    else:
                        missing_files.append(str(results_file))
        
        coverage_stats = {
            'total_expected': total_expected,
            'existing_count': len(existing_files),
            'missing_count': len(missing_files),
            'coverage_percentage': (len(existing_files) / total_expected * 100) if total_expected > 0 else 0,
            'existing_files': existing_files,
            'missing_files': missing_files
        }
        
        return coverage_stats


class SchemaMappingMethod:
    """Base class for schema mapping methods."""
    
    def __init__(self, name: str, model: str = "granite"):
        self.name = name
        self.model = model
    
    def validate_result_format(self, data: Dict[str, Any]) -> bool:
        """Validate that the result file has the expected format."""
        # Only 'results' is required, 'metadata' is optional
        if 'results' not in data:
            return False
        
        # Check if results is a list
        if not isinstance(data['results'], list):
            return False
        
        # Check if each result has required fields
        for result in data['results']:
            if not isinstance(result, dict):
                return False
            required_result_fields = ['sample_id', 'predicted_mapping', 'accuracy']
            if not all(field in result for field in required_result_fields):
                return False
        
        return True


# Remove all the mock method classes since we're loading from files now
# MapReflectMethod, SMTMethod, GNNMethod, AnyToolMethod, ReActMethod are no longer needed


class EvaluationMetrics:
    """Calculate evaluation metrics for schema mapping."""
    
    @staticmethod
    def calculate_accuracy(ground_truth: Dict[str, Any], predicted: Dict[str, Any]) -> float:
        """
        Calculate accuracy for schema mapping.
        Returns: accuracy (proportion of correct mappings)
        """
        if not ground_truth:
            return 0.0
        
        # Convert to sets for easier comparison
        gt_pairs = set(ground_truth.items())
        pred_pairs = set(predicted.items())
        
        # True positives: correct mappings
        correct_mappings = len(gt_pairs.intersection(pred_pairs))
        
        # Accuracy: correct mappings / total ground truth mappings
        accuracy = correct_mappings / len(gt_pairs) if gt_pairs else 0.0
        
        return accuracy


class SchemaMappingEvaluator:
    """Main evaluation framework for schema mapping experiments."""
    
    def __init__(self, base_dir: Path = Path(".")):
        self.loader = DatasetLoader()
        self.result_loader = ResultFileLoader(base_dir)
        self.metrics_calculator = EvaluationMetrics()
        self.results: List[EvaluationResult] = []
        
        # Method names (no longer instantiate method classes)
        self.methods = ['SMT', 'AnyTool', 'ReAct', 'GNN', 'Map-Reflect']
        
        # Models to evaluate
        self.models = ['granite', 'llama', 'qwen']
        
        # Datasets to evaluate (removed AnyToolBench - REST API dataset)
        self.datasets = ['ITOps', 'NL2GQL', 'GraphQLRestBench']
    
    async def run_comprehensive_evaluation(self, 
                                         datasets: List[str] = None,
                                         methods: List[str] = None,
                                         models: List[str] = None,
                                         max_samples_per_dataset: int = 50,
                                         output_dir: Path = None) -> List[EvaluationResult]:
        """Load results from JSON files and process them."""
        
        datasets = datasets or self.datasets
        methods = methods or self.methods
        models = models or self.models
        
        logger.info(f"Starting comprehensive evaluation from result files")
        logger.info(f"Datasets: {datasets}")
        logger.info(f"Methods: {methods}")
        logger.info(f"Models: {models}")
        
        # Check file coverage first
        coverage_stats = self.result_loader.check_file_coverage(methods, datasets, models)
        
        logger.info(f"\n📁 File Coverage Summary:")
        logger.info(f"Total expected files: {coverage_stats['total_expected']}")
        logger.info(f"Existing files: {coverage_stats['existing_count']}")
        logger.info(f"Missing files: {coverage_stats['missing_count']}")
        logger.info(f"Coverage: {coverage_stats['coverage_percentage']:.1f}%")
        
        if coverage_stats['missing_files']:
            logger.info(f"\n❌ Missing result files:")
            for missing_file in coverage_stats['missing_files']:
                logger.info(f"   {missing_file}")
        
        if coverage_stats['existing_files']:
            logger.info(f"\n✅ Found result files:")
            for existing_file in coverage_stats['existing_files']:
                logger.info(f"   {existing_file}")
        
        # Load existing processed results if available
        existing_results = []
        if output_dir and (output_dir / "detailed_results.csv").exists() and not hasattr(self, '_force_regenerate'):
            existing_results = self._load_existing_processed_results(output_dir)
        
        # Create a set of completed combinations for quick lookup
        completed_combinations = set()
        for result in existing_results:
            key = f"{result.dataset}_{result.method}_{result.model}"
            completed_combinations.add(key)
        
        all_results = existing_results.copy()
        
        # Process result files
        for dataset_name in datasets:
            for method_name in methods:
                for model_name in models:
                    combination_key = f"{dataset_name}_{method_name}_{model_name}"
                    
                    if combination_key in completed_combinations:
                        logger.info(f"  Skipping {combination_key} - already processed")
                        continue
                    
                    logger.info(f"Processing {method_name} with {model_name} on {dataset_name}")
                    
                    # Load result file
                    result_data = self.result_loader.load_method_results(method_name, dataset_name, model_name)
                    
                    if result_data is None:
                        logger.warning(f"  No results found for {combination_key}")
                        continue
                    
                    # Validate result format
                    method_obj = SchemaMappingMethod(method_name, model_name)
                    if not method_obj.validate_result_format(result_data):
                        logger.error(f"  Invalid result format in {combination_key}")
                        continue
                    
                    # Process the results
                    method_results = self._process_result_file(
                        result_data, dataset_name, method_name, model_name
                    )
                    all_results.extend(method_results)
                    
                    # Save incremental results
                    if output_dir:
                        self.results = all_results
                        self._save_incremental_results(output_dir)
        
        self.results = all_results
        logger.info(f"\n✅ Evaluation completed. Total results: {len(all_results)}")
        
        # Final coverage report
        processed_combinations = set()
        for result in all_results:
            key = f"{result.dataset}_{result.method}_{result.model}"
            processed_combinations.add(key)
        
        total_expected_combinations = len(datasets) * len(methods) * len(models)
        logger.info(f"\n📊 Processing Summary:")
        logger.info(f"Expected combinations: {total_expected_combinations}")
        logger.info(f"Processed combinations: {len(processed_combinations)}")
        logger.info(f"Processing rate: {len(processed_combinations)/total_expected_combinations*100:.1f}%")
        
        return all_results
    
    def _save_incremental_results(self, output_dir: Path):
        """Save results incrementally to avoid losing progress."""
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        df = pd.DataFrame([asdict(result) for result in self.results])
        csv_file = output_dir / "detailed_results.csv"
        df.to_csv(csv_file, index=False)
        
        logger.info(f"Saved {len(self.results)} results to {csv_file}")
    
    def _load_existing_processed_results(self, output_dir: Path) -> List[EvaluationResult]:
        """Load existing processed results from CSV."""
        try:
            existing_df = pd.read_csv(output_dir / "detailed_results.csv")
            logger.info(f"Found existing CSV with columns: {list(existing_df.columns)}")
            
            existing_results = []
            for _, row in existing_df.iterrows():
                try:
                    row_dict = row.to_dict()
                    
                    # Handle dictionary columns - use json.loads for safety
                    for dict_col in ['ground_truth_mapping', 'predicted_mapping']:
                        if dict_col in row_dict:
                            if isinstance(row_dict[dict_col], str):
                                try:
                                    row_dict[dict_col] = json.loads(row_dict[dict_col].replace("'", '"'))
                                except:
                                    try:
                                        row_dict[dict_col] = eval(row_dict[dict_col])
                                    except:
                                        row_dict[dict_col] = {}
                            elif pd.isna(row_dict[dict_col]):
                                row_dict[dict_col] = {}
                    
                    # Handle optional fields
                    for optional_field in ['error_message']:
                        if optional_field not in row_dict or pd.isna(row_dict[optional_field]):
                            row_dict[optional_field] = None
                    
                    # Convert success to boolean
                    if 'success' in row_dict:
                        if isinstance(row_dict['success'], str):
                            row_dict['success'] = row_dict['success'].lower() == 'true'
                        else:
                            row_dict['success'] = bool(row_dict['success'])
                    
                    # Ensure all required fields are present with defaults
                    required_fields = {
                        'dataset': 'unknown',
                        'method': 'unknown', 
                        'model': 'unknown',
                        'sample_id': 'unknown',
                        'source_schema': '',
                        'target_schema': '',
                        'accuracy': 0.0,
                        'execution_time': 0.0,
                        'success': False
                    }
                    
                    for field, default_value in required_fields.items():
                        if field not in row_dict or pd.isna(row_dict[field]):
                            row_dict[field] = default_value
                    
                    existing_results.append(EvaluationResult(**row_dict))
                    
                except Exception as e:
                    logger.warning(f"Could not parse row: {e}")
                    continue
            
            logger.info(f"Successfully loaded {len(existing_results)} existing results")
            return existing_results
            
        except Exception as e:
            logger.warning(f"Could not load existing results: {e}")
            return []
    
    def _process_result_file(self, result_data: Dict[str, Any], dataset: str, method: str, model: str) -> List[EvaluationResult]:
        """Process a result file and convert to EvaluationResult objects."""
        results = []
        
        # Metadata is optional
        metadata = result_data.get('metadata', {})
        execution_time = metadata.get('execution_time', 0.0)
        
        for result_entry in result_data['results']:
            try:
                sample_id = result_entry['sample_id']
                predicted_mapping = result_entry['predicted_mapping']
                accuracy = result_entry['accuracy']
                
                # Get optional fields
                ground_truth_mapping = result_entry.get('ground_truth_mapping', {})
                source_schema = result_entry.get('source_schema', '')
                target_schema = result_entry.get('target_schema', '')
                error_message = result_entry.get('error_message')
                success = result_entry.get('success', True)
                
                # Truncate schemas for storage
                if len(source_schema) > 200:
                    source_schema = source_schema[:200] + "..."
                if len(target_schema) > 200:
                    target_schema = target_schema[:200] + "..."
                
                result = EvaluationResult(
                    dataset=dataset,
                    method=method,
                    model=model,
                    sample_id=sample_id,
                    source_schema=source_schema,
                    target_schema=target_schema,
                    ground_truth_mapping=ground_truth_mapping,
                    predicted_mapping=predicted_mapping,
                    accuracy=accuracy,
                    execution_time=execution_time,
                    error_message=error_message,
                    success=success
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing result entry {result_entry.get('sample_id', 'unknown')}: {e}")
                continue
        
        logger.info(f"  Processed {len(results)} results from {dataset}_{method}_{model}")
        return results
    
    # Remove the old _evaluate_method_on_dataset method since we're not running methods anymore
    # Results are loaded from files instead
    
    def generate_latex_tables(self, output_dir: Path, generate_latex: bool = True) -> Dict[str, Any]:
        """Generate LaTeX tables for AAAI paper."""
        if not self.results:
            logger.error("No results to generate tables from")
            return {}
        
        output_dir.mkdir(exist_ok=True)
        
        # Convert results to DataFrame for analysis
        df = pd.DataFrame([asdict(result) for result in self.results])
        
        # Calculate aggregate statistics
        stats = self._calculate_aggregate_statistics(df)
        
        # Always save detailed results and statistics
        df.to_csv(output_dir / "detailed_results.csv", index=False)
        
        with open(output_dir / "evaluation_statistics.json", 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        # Generate LaTeX tables only if requested
        if generate_latex:
            logger.info("Generating LaTeX tables...")
            self._generate_main_results_table(stats, output_dir / "main_results.tex")
            self._generate_method_comparison_table(stats, output_dir / "method_comparison.tex")
            self._generate_dataset_analysis_table(stats, output_dir / "dataset_analysis.tex")
            self._generate_model_comparison_table(stats, output_dir / "model_comparison.tex")
            logger.info(f"LaTeX tables saved to {output_dir}")
        else:
            logger.info("Skipping LaTeX table generation (use --generate-latex to enable)")
        
        logger.info(f"Results and statistics saved to {output_dir}")
        return stats
    
    def _calculate_aggregate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate aggregate statistics from results - ONLY ACCURACY."""
        
        # Convert ground_truth_mapping and predicted_mapping from string back to dict if needed
        for col in ['ground_truth_mapping', 'predicted_mapping']:
            if col in df.columns and df[col].dtype == 'object':
                try:
                    df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)
                except:
                    logger.warning(f"Could not parse {col} column")
        
        stats = {
            'overall': {
                'total_evaluations': len(df),
                'successful_evaluations': len(df[df['success'] == True]),
                'avg_accuracy': df['accuracy'].mean() if len(df) > 0 else 0.0,
                'avg_execution_time': df['execution_time'].mean() if len(df) > 0 else 0.0
            },
            'by_method': {},
            'by_dataset': {},
            'by_model': {},
            'by_method_dataset': {},
            'by_method_model': {}
        }
        
        # By method
        for method in df['method'].unique():
            method_df = df[df['method'] == method]
            method_successful = method_df[method_df['success'] == True]
            
            stats['by_method'][method] = {
                'count': len(method_df),
                'success_rate': len(method_successful) / len(method_df) if len(method_df) > 0 else 0.0,
                'avg_accuracy': method_df['accuracy'].mean() if len(method_df) > 0 else 0.0,
                'std_accuracy': method_df['accuracy'].std() if len(method_df) > 0 else 0.0,
                'avg_execution_time': method_df['execution_time'].mean() if len(method_df) > 0 else 0.0
            }
        
        # By dataset
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            dataset_successful = dataset_df[dataset_df['success'] == True]
            
            stats['by_dataset'][dataset] = {
                'count': len(dataset_df),
                'success_rate': len(dataset_successful) / len(dataset_df) if len(dataset_df) > 0 else 0.0,
                'avg_accuracy': dataset_df['accuracy'].mean() if len(dataset_df) > 0 else 0.0
            }
        
        # By model
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            model_successful = model_df[model_df['success'] == True]
            
            stats['by_model'][model] = {
                'count': len(model_df),
                'success_rate': len(model_successful) / len(model_df) if len(model_df) > 0 else 0.0,
                'avg_accuracy': model_df['accuracy'].mean() if len(model_df) > 0 else 0.0
            }
        
        # By method-dataset combination
        for method in df['method'].unique():
            for dataset in df['dataset'].unique():
                method_dataset_df = df[(df['method'] == method) & (df['dataset'] == dataset)]
                if len(method_dataset_df) > 0:
                    key = f"{method}_{dataset}"
                    stats['by_method_dataset'][key] = {
                        'count': len(method_dataset_df),
                        'avg_accuracy': method_dataset_df['accuracy'].mean()
                    }
        
        # By method-model combination
        for method in df['method'].unique():
            for model in df['model'].unique():
                method_model_df = df[(df['method'] == method) & (df['model'] == model)]
                if len(method_model_df) > 0:
                    key = f"{method}_{model}"
                    stats['by_method_model'][key] = {
                        'count': len(method_model_df),
                        'avg_accuracy': method_model_df['accuracy'].mean()
                    }
        
        return stats
    
    def _generate_main_results_table(self, stats: Dict[str, Any], output_file: Path):
        """Generate main results table comparing all methods."""
        latex_content = """% Main Results Table for AAAI Paper
% Map-Reflect: GraphQL Schema Mapping using Self Reflection

\\begin{table*}[htbp]
\\centering
\\caption{Schema Mapping Performance Comparison Across Methods and Models}
\\label{tab:main_results}
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Method} & \\textbf{Model} & \\textbf{Accuracy} & \\textbf{Success Rate} & \\textbf{Samples} \\\\
\\midrule
"""
        
        # Get methods in specified order: SMT, AnyTool, ReAct, GNN, Map-Reflect
        methods = ['SMT', 'AnyTool', 'ReAct', 'GNN', 'Map-Reflect']
        models = ['granite', 'llama', 'qwen']
        
        for method in methods:
            for i, model in enumerate(models):
                key = f"{method}_{model}"
                
                if key in stats['by_method_model']:
                    method_stats = stats['by_method_model'][key]
                    accuracy = method_stats['avg_accuracy']
                    count = method_stats['count']
                    
                    # Get success rate from method stats
                    if method in stats['by_method']:
                        success_rate = stats['by_method'][method]['success_rate']
                    else:
                        success_rate = 0.0
                    
                    # Format method name only for first model
                    method_name = method if i == 0 else ""
                    
                    latex_content += f"{method_name} & {model} & {accuracy:.3f} & {success_rate:.3f} & {count} \\\\\n"
                else:
                    # Missing result - mark as TBA
                    method_name = method if i == 0 else ""
                    latex_content += f"{method_name} & {model} & TBA & TBA & TBA \\\\\n"
            
            # Add separator between methods
            if method != methods[-1]:
                latex_content += "\\midrule\n"
        
        latex_content += """\\bottomrule
\\end{tabular}%
}
\\end{table*}

% Table notes and explanation
The main results table presents the performance of all evaluated schema mapping methods across different language models. The methods are ordered as: SMT (Schema Mapping Toolkit), AnyTool, ReAct, GNN, and Map-Reflect (our proposed method). Accuracy measures the proportion of correctly identified schema mappings. Success Rate indicates the percentage of evaluations that completed without errors. Samples shows the number of schema pairs evaluated per method-model combination. Results marked as TBA (To Be Added) indicate experiments currently in progress or methods yet to be implemented.

"""
        
        with open(output_file, 'w') as f:
            f.write(latex_content)
    
    def _generate_method_comparison_table(self, stats: Dict[str, Any], output_file: Path):
        """Generate method comparison table across datasets."""
        latex_content = """% Method Comparison Across Datasets
% Map-Reflect: GraphQL Schema Mapping using Self Reflection

\\begin{table*}[htbp]
\\centering
\\caption{Method Performance Across Different Datasets}
\\label{tab:method_comparison}
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{lccccc}
\\toprule
\\textbf{Method} & \\textbf{ITOps} & \\textbf{NL2GQL} & \\textbf{GraphQLRestBench} & \\textbf{Average} & \\textbf{Rank} \\\\
\\midrule
"""
        
        methods = ['SMT', 'AnyTool', 'ReAct', 'GNN', 'Map-Reflect']
        datasets = ['ITOps', 'NL2GQL', 'GraphQLRestBench']
        
        method_averages = {}
        
        for method in methods:
            dataset_scores = []
            row_content = f"{method}"
            
            for dataset in datasets:
                key = f"{method}_{dataset}"
                if key in stats['by_method_dataset']:
                    accuracy = stats['by_method_dataset'][key]['avg_accuracy']
                    dataset_scores.append(accuracy)
                    row_content += f" & {accuracy:.3f}"
                else:
                    row_content += " & TBA"
            
            # Calculate average (only from available scores)
            if dataset_scores:
                avg_score = sum(dataset_scores) / len(dataset_scores)
                method_averages[method] = avg_score
                row_content += f" & {avg_score:.3f}"
            else:
                method_averages[method] = 0.0
                row_content += " & TBA"
            
            # Rank will be filled after calculating all averages
            row_content += " & TBA"
            latex_content += row_content + " \\\\\n"
        
        latex_content += """\\bottomrule
\\end{tabular}%
}
\\end{table*}

% Dataset-specific analysis
This table compares method performance across different GraphQL datasets, highlighting the adaptability of each approach to various schema types. ITOps represents IT operations monitoring schemas, NL2GQL focuses on natural language to GraphQL translation scenarios, and GraphQLRestBench provides GraphQL-REST API mapping challenges. The Average column shows overall method performance, while Rank indicates relative standing (1=best). Map-Reflect's self-reflection mechanism is expected to show superior adaptability across diverse schema types.

"""
        
        with open(output_file, 'w') as f:
            f.write(latex_content)
    
    def _generate_dataset_analysis_table(self, stats: Dict[str, Any], output_file: Path):
        """Generate dataset analysis table."""
        latex_content = """% Dataset Analysis Table
% Map-Reflect: GraphQL Schema Mapping using Self Reflection

\\begin{table}[htbp]
\\centering
\\caption{Dataset Characteristics and Overall Performance}
\\label{tab:dataset_analysis}
\\begin{tabular}{lccc}
\\toprule
\\textbf{Dataset} & \\textbf{Schema Pairs} & \\textbf{Avg Accuracy} & \\textbf{Complexity} \\\\
\\midrule
"""
        
        datasets = ['ITOps', 'NL2GQL', 'GraphQLRestBench']
        complexity_map = {
            'ITOps': 'Monitoring schemas',
            'NL2GQL': 'NL-to-GraphQL', 
            'GraphQLRestBench': 'GraphQL-REST mapping'
        }
        
        for dataset in datasets:
            if dataset in stats['by_dataset']:
                dataset_stats = stats['by_dataset'][dataset]
                count = dataset_stats['count'] // 5  # Assuming 5 methods per dataset
                accuracy = dataset_stats['avg_accuracy']
                complexity = complexity_map.get(dataset, 'Unknown')
                
                latex_content += f"{dataset} & {count} & {accuracy:.3f} & {complexity} \\\\\n"
            else:
                latex_content += f"{dataset} & TBA & TBA & {complexity_map.get(dataset, 'Unknown')} \\\\\n"
        
        latex_content += """\\bottomrule
\\end{tabular}
\\end{table}

% Dataset characteristics explanation
This table summarizes the characteristics of each evaluation dataset. Schema Pairs indicates the number of source-target schema pairs evaluated. Average Accuracy shows overall performance across all methods for each dataset. Complexity describes the type of mapping challenge each dataset presents. These metrics help understand dataset-specific challenges and method performance patterns across different GraphQL schema mapping scenarios.

"""
        
        with open(output_file, 'w') as f:
            f.write(latex_content)
    
    def _generate_model_comparison_table(self, stats: Dict[str, Any], output_file: Path):
        """Generate model comparison table."""
        latex_content = """% Model Comparison Table
% Map-Reflect: GraphQL Schema Mapping using Self Reflection

\\begin{table}[htbp]
\\centering
\\caption{Language Model Performance in Schema Mapping}
\\label{tab:model_comparison}
\\begin{tabular}{lcc}
\\toprule
\\textbf{Model} & \\textbf{Avg Accuracy} & \\textbf{Success Rate} \\\\
\\midrule
"""
        
        models = ['granite', 'llama', 'qwen']
        
        for model in models:
            if model in stats['by_model']:
                model_stats = stats['by_model'][model]
                accuracy = model_stats['avg_accuracy']
                success_rate = model_stats['success_rate']
                
                latex_content += f"{model.title()} & {accuracy:.3f} & {success_rate:.3f} \\\\\n"
            else:
                latex_content += f"{model.title()} & TBA & TBA \\\\\n"
        
        latex_content += """\\bottomrule
\\end{tabular}
\\end{table}

% Model comparison analysis
This table evaluates the performance of different language models in schema mapping tasks. Granite represents IBM's foundation model, Llama shows Meta's open-source model performance, and Qwen demonstrates Alibaba's multilingual model capabilities. The comparison helps identify which language models are most effective for schema understanding and mapping generation across different architectural approaches and training paradigms.

"""
        
        with open(output_file, 'w') as f:
            f.write(latex_content)

async def main():
    """Main entry point for schema mapping evaluation."""
    parser = argparse.ArgumentParser(description="Schema Mapping Evaluation for AAAI Paper")
    parser.add_argument("--datasets", nargs='+', default=['ITOps', 'NL2GQL'], 
                       help="Datasets to evaluate on")
    parser.add_argument("--methods", nargs='+', default=['SMT', 'AnyTool', 'Map-Reflect'], 
                       help="Methods to evaluate")
    parser.add_argument("--models", nargs='+', default=['granite', 'llama'], 
                       help="Models to evaluate")
    parser.add_argument("--max-samples", type=int, default=50, 
                       help="Maximum samples per dataset")
    parser.add_argument("--output-dir", default="./results", 
                       help="Output directory for results and tables")
    parser.add_argument("--quick", action="store_true", 
                       help="Quick evaluation with minimal samples")
    parser.add_argument("--force-regenerate", action="store_true",
                       help="Force regeneration of all results (ignore existing)")
    parser.add_argument("--clean", action="store_true",
                       help="Clean output directory before starting")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    parser.add_argument("--generate-latex", action="store_true",
                       help="Generate LaTeX tables for paper")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    if args.quick:
        args.max_samples = 10
        args.datasets = ['ITOps']
        args.methods = ['SMT', 'Map-Reflect']
        args.models = ['granite']
        logger.info("Quick evaluation mode: using minimal configuration")
    
    logger.info(f"Schema Mapping Evaluation for AAAI Paper")
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Methods: {args.methods}")
    logger.info(f"Models: {args.models}")
    logger.info(f"Max samples per dataset: {args.max_samples}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Initialize evaluator with base directory for result files
    evaluator = SchemaMappingEvaluator(base_dir=Path("."))
    
    # Handle output directory
    output_dir = Path(args.output_dir)
    
    if args.clean and output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
        logger.info(f"Cleaned output directory: {output_dir}")
    
    if args.force_regenerate:
        # Remove existing results to force regeneration
        results_file = output_dir / "detailed_results.csv"
        if results_file.exists():
            results_file.unlink()
            logger.info("Forcing regeneration: removed existing results file")
        evaluator._force_regenerate = True
    
    # Run comprehensive evaluation
    start_time = time.time()
    results = await evaluator.run_comprehensive_evaluation(
        datasets=args.datasets,
        methods=args.methods,
        models=args.models,
        max_samples_per_dataset=args.max_samples,
        output_dir=output_dir
    )
    total_time = time.time() - start_time
    
    if not results:
        logger.error("No results collected")
        sys.exit(1)
    
    # Generate LaTeX tables and analysis
    output_dir = Path(args.output_dir)
    stats = evaluator.generate_latex_tables(output_dir, generate_latex=args.generate_latex)
    
    # Print summary
    logger.info(f"\n🎉 Schema Mapping Evaluation Completed!")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Total evaluations: {len(results)}")
    
    overall = stats['overall']
    logger.info(f"\n📈 Overall Results:")
    logger.info(f"Success Rate: {overall['successful_evaluations']}/{overall['total_evaluations']}")
    logger.info(f"Average Accuracy: {overall['avg_accuracy']:.3f}")
    
    logger.info(f"\n🤖 Method Performance:")
    for method, metrics in stats['by_method'].items():
        logger.info(f"{method}: Accuracy={metrics['avg_accuracy']:.3f}")
    
    logger.info(f"\n📊 Dataset Performance:")
    for dataset, metrics in stats['by_dataset'].items():
        logger.info(f"{dataset}: Accuracy={metrics['avg_accuracy']:.3f}")
    
    logger.info(f"\n🔧 Model Performance:")
    for model, metrics in stats['by_model'].items():
        logger.info(f"{model}: Accuracy={metrics['avg_accuracy']:.3f}")
    
    logger.info(f"\n📁 Results saved to: {output_dir}")
    logger.info(f"Data files:")
    logger.info(f"  - detailed_results.csv: Raw evaluation data")
    logger.info(f"  - evaluation_statistics.json: Aggregated statistics")
    
    if args.generate_latex:
        logger.info(f"LaTeX tables:")
        logger.info(f"  - main_results.tex: Main performance comparison")
        logger.info(f"  - method_comparison.tex: Method comparison across datasets")
        logger.info(f"  - dataset_analysis.tex: Dataset characteristics")
        logger.info(f"  - model_comparison.tex: Language model comparison")
    else:
        logger.info(f"Use --generate-latex to create LaTeX tables for paper")


if __name__ == "__main__":
    asyncio.run(main())
