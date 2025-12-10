"""
Evaluation Main Script

This script evaluates and compares the performance of different AI models on Swiss professional exams.
It aggregates results across multiple exams and judges, and generates comparison plots.
"""

import os
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class EvaluationPipeline:
    """Pipeline for evaluating and comparing model performance across exams."""
    
    def __init__(self, config_path: str):
        """Initialize the evaluation pipeline.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.eval_config = self.config.get('evaluation', {})
        self.benchmarked_data_dir = Path(self.config['benchmarked_data_dir'])
        self.eval_output_dir = Path(self.config['eval_data_dir'])
        
        # Create timestamped output directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.eval_output_dir / self.timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store metadata about which benchmarking runs were used
        self.used_benchmarking_runs = {}
        
    def find_latest_benchmarking_run(self, profession: str, exam_number: str, model: str) -> Tuple[str, str]:
        """Find the latest benchmarking run for a given exam and model across all processing runs.
        
        Args:
            profession: Profession name
            exam_number: Exam number (folder name)
            model: Model name (to find the specific model folder)
            
        Returns:
            Tuple of (processing_timestamp, benchmark_timestamp) or (None, None)
        """
        exam_benchmarked_dir = self.benchmarked_data_dir / profession / exam_number
        if not exam_benchmarked_dir.exists():
            return None, None
        
        # Get all processing timestamps
        processing_timestamps = [d.name for d in exam_benchmarked_dir.iterdir() if d.is_dir()]
        if not processing_timestamps:
            return None, None
        
        # Find the latest benchmarking run for THIS SPECIFIC MODEL across all processing runs
        latest_benchmark_timestamp = None
        latest_processing_timestamp = None
        
        model_safe = model.replace('/', '__')
        # Strict matching: The folder name must match the config model name (sanitized)
        # If user wants RAG, they must put "model_name_rag" in the config.
        model_folder_name = f"model={model_safe}"
        
        for proc_ts in processing_timestamps:
            proc_dir = exam_benchmarked_dir / proc_ts
            model_dir = proc_dir / model_folder_name
            
            if model_dir.exists() and model_dir.is_dir():
                # In the new structure, benchmark timestamps are inside the model folder
                benchmark_timestamps = [d.name for d in model_dir.iterdir() if d.is_dir()]
                
                for bench_ts in benchmark_timestamps:
                    if latest_benchmark_timestamp is None or bench_ts > latest_benchmark_timestamp:
                        latest_benchmark_timestamp = bench_ts
                        latest_processing_timestamp = proc_ts
        
        return latest_processing_timestamp, latest_benchmark_timestamp
    
    def load_graded_results(self, profession: str, exam_number: str, processing_timestamp: str,
                           benchmark_timestamp: str, model: str, judge: str) -> Dict:
        """Load graded results for a specific model and judge."""
        # Convert slashes to double underscores for filesystem
        model_safe = model.replace('/', '__')
        judge_safe = judge.replace('/', '__')
        
        # Base path to processed run
        base_path = (self.benchmarked_data_dir / profession / exam_number / 
                    processing_timestamp)
        
        # Construct path using strict structure: .../processed_ts/model/benchmark_ts/judge/graded.json
        model_folder = f"model={model_safe}"
        graded_file = (base_path / model_folder / benchmark_timestamp / 
                      f"judge={judge_safe}" / "graded_answers.json")
        
        if not graded_file.exists():
            print(f"Warning: Graded file not found: {graded_file}")
            return None
        
        with open(graded_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def aggregate_results(self) -> Dict[str, Dict]:
        """Aggregate results across all specified exams, models, and judges."""
        professions = self.eval_config.get('professions', [])
        exam_numbers = self.eval_config.get('exam_numbers')
        models = self.eval_config.get('models', [])
        judges = self.eval_config.get('judges', [])
        
        # Initialize results structure
        model_results = {model: {'percentages': [], 'std_devs': [], 'metadata': []} for model in models}
        
        # Iterate through all exam combinations
        for profession in professions:
            profession_dir = self.benchmarked_data_dir / profession
            if not profession_dir.exists():
                continue
                
            # Iterate over numbered folders
            for exam_num_dir in profession_dir.iterdir():
                if not exam_num_dir.is_dir():
                    continue
                
                exam_number = exam_num_dir.name
                
                # Filter by exam number if specified
                if exam_numbers and exam_numbers != "all" and exam_number not in exam_numbers:
                    continue
                
                # Aggregate across judges for each model
                for model in models:
                    # Find latest benchmarking run FOR THIS MODEL
                    processing_timestamp, benchmark_timestamp = self.find_latest_benchmarking_run(
                        profession, exam_number, model
                    )
                    
                    if not processing_timestamp or not benchmark_timestamp:
                        continue

                    # Store which benchmarking run was used
                    exam_key = f"{profession}/{exam_number}/{model}"
                    self.used_benchmarking_runs[exam_key] = {
                        'processing_timestamp': processing_timestamp,
                        'benchmark_timestamp': benchmark_timestamp
                    }

                    for judge in judges:
                        graded_data = self.load_graded_results(
                            profession, exam_number, processing_timestamp,
                            benchmark_timestamp, model, judge
                        )
                        
                        if graded_data is None:
                            continue
                        
                        # Extract stats from new grading_summary structure
                        grading_summary = graded_data.get('grading_summary', {})
                        aggregation = grading_summary.get('aggregation', {})
                        
                        # Fallback for old structure if 'aggregation' is missing
                        if not aggregation:
                            percentage = grading_summary.get('percentage', 0.0)
                            std_dev = 0.0
                        else:
                            percentage = aggregation.get('average_percentage', 0.0)
                            std_dev = aggregation.get('std_dev_percentage', 0.0)
                        
                        model_results[model]['percentages'].append(percentage)
                        model_results[model]['std_devs'].append(std_dev)
                        model_results[model]['metadata'].append({
                            'profession': profession,
                            'exam_number': exam_number,
                            'processing_timestamp': processing_timestamp,
                            'benchmark_timestamp': benchmark_timestamp,
                            'judge': judge,
                            'grading_summary': grading_summary
                        })
        
        return model_results
    
    def create_plot(self, model_results: Dict[str, Dict]) -> str:
        """Create a bar plot comparing model performance.
        
        Args:
            model_results: Aggregated results from aggregate_results()
            
        Returns:
            Path to the saved plot
        """
        # Set publication-ready style
        plt.style.use('seaborn-v0_8-whitegrid')
        matplotlib.rcParams.update({
            'font.size': 14,
            'axes.labelsize': 16,
            'axes.titlesize': 18,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'figure.figsize': (14, 9),
            'axes.grid': True,
            'grid.alpha': 0.3
        })
        
        model_names = []
        avg_percentages = []
        std_percentages = []
        
        # Determine if we are aggregating over multiple exams
        all_exams = set()
        for model, data in model_results.items():
            for meta in data['metadata']:
                all_exams.add((meta['profession'], meta['exam_number']))
        
        is_multi_exam = len(all_exams) > 1
        
        # Prepare data for plotting
        for model, data in model_results.items():
            if data['percentages']:
                # Extract just the model name without provider
                display_name = model.split('/')[-1] if '/' in model else model
                model_names.append(display_name)
                
                if is_multi_exam:
                    # Group by exam first to handle multiple judges per exam correctly
                    exam_scores = {} # (prof, num) -> [scores]
                    for score, meta in zip(data['percentages'], data['metadata']):
                        key = (meta['profession'], meta['exam_number'])
                        if key not in exam_scores:
                            exam_scores[key] = []
                        exam_scores[key].append(score)
                    
                    # Calculate average score for each exam (averaging over judges if multiple)
                    avg_score_per_exam = [np.mean(scores) for scores in exam_scores.values()]
                    
                    # Overall average across exams
                    avg_percentages.append(np.mean(avg_score_per_exam))
                    # Standard deviation across exams (Sample STD)
                    if len(avg_score_per_exam) > 1:
                        std_percentages.append(np.std(avg_score_per_exam, ddof=1))
                    else:
                        std_percentages.append(0.0)
                else:
                    # Single exam case
                    if len(data['percentages']) == 1:
                        # Single judge run(s) -> use the std dev from the runs
                        avg_percentages.append(data['percentages'][0])
                        std_percentages.append(data['std_devs'][0])
                    else:
                        # Multiple judges -> use std dev across judges (Sample STD)
                        avg_percentages.append(np.mean(data['percentages']))
                        if len(data['percentages']) > 1:
                            std_percentages.append(np.std(data['percentages'], ddof=1))
                        else:
                            std_percentages.append(0.0)
        
        if not model_names:
            print("No data to plot.")
            return ""

        # Sort models if requested in config
        if self.eval_config.get('sort', False):
            # Zip, sort by average percentage (descending), and unzip
            combined = sorted(zip(model_names, avg_percentages, std_percentages), 
                            key=lambda x: x[1], reverse=True)
            model_names, avg_percentages, std_percentages = zip(*combined)
            # Convert back to lists
            model_names = list(model_names)
            avg_percentages = list(avg_percentages)
            std_percentages = list(std_percentages)

        # Create the bar plot
        fig, ax = plt.subplots()
        x_pos = np.arange(len(model_names))
        
        # Plot bars with error bars
        bars = ax.bar(x_pos, avg_percentages, yerr=std_percentages, 
                     capsize=10, alpha=0.85, color='#4C72B0', 
                     edgecolor='black', linewidth=1.5, width=0.6)
        
        # Customize axes
        ax.set_ylabel('Score (%)', fontweight='bold', labelpad=15)
        ax.set_xlabel('Candidate Model', fontweight='bold', labelpad=15)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylim(0, 100)
        
        # Add value labels
        for i, (bar, avg, std) in enumerate(zip(bars, avg_percentages, std_percentages)):
            height = bar.get_height()
            label_y = height + std + 2 if height + std + 2 < 95 else height - 5
            color = 'black' if height + std + 2 < 95 else 'white'
            
            ax.text(bar.get_x() + bar.get_width()/2., label_y,
                   f'{avg:.1f}%\n(±{std:.1f})',
                   ha='center', va='bottom', fontweight='bold', color=color)
        
        # Generate Title
        professions = self.eval_config.get('professions', [])
        judges = self.eval_config.get('judges', [])
        
        prof_str = ', '.join(professions)
        judge_names = [j.split('/')[-1] if '/' in j else j for j in judges]
        judge_str = ', '.join(judge_names)
        
        # Get number of judge runs from the first result
        n_runs = "N/A"
        models = self.eval_config.get('models', [])
        if models and models[0] in model_results and model_results[models[0]]['metadata']:
            meta = model_results[models[0]]['metadata'][0]
            if 'grading_summary' in meta and 'judge_runs' in meta['grading_summary']:
                n_runs = len(meta['grading_summary']['judge_runs'])
        
        # Add counts to title
        if is_multi_exam:
            prof_str += f" ({len(all_exams)} exams)"
        
        judge_str += f" ({n_runs} runs)"
        
        title = (f"Model Performance Comparison\n"
                 f"Exam: {prof_str}\n"
                 f"Judge: {judge_str}")
        
        ax.set_title(title, fontweight='bold', pad=20)
        
        # Add caption
        if is_multi_exam:
            caption = f"Error bars represent standard deviation across {len(all_exams)} exams."
        else:
            # Check if we have multiple judges
            if len(judges) > 1:
                 caption = f"Error bars represent standard deviation across {len(judges)} judges."
            else:
                 caption = f"Error bars represent standard deviation across {n_runs} judge runs."
                 
        fig.text(0.5, 0.02, caption, ha='center', fontsize=12, style='italic')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        
        # Save
        plot_filename = f"model_comparison_{self.timestamp}.png"
        plot_path = self.output_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved to: {plot_path}")
        return str(plot_path)
    
    def save_metadata(self):
        """Save metadata about which benchmarking runs were used."""
        metadata = {
            'evaluation_timestamp': self.timestamp,
            'configuration': self.eval_config,
            'benchmarking_runs_used': self.used_benchmarking_runs
        }
        
        metadata_file = self.output_dir / "evaluation_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Metadata saved to: {metadata_file}")
    
    def run(self):
        """Execute the complete evaluation pipeline."""
        print("="*80)
        print("EVALUATION PIPELINE")
        print("="*80)
        print(f"Output directory: {self.output_dir}")
        print()
        
        # Aggregate results
        print("Aggregating results...")
        model_results = self.aggregate_results()
        
        # Print summary
        print("\nResults Summary:")
        for model, data in model_results.items():
            if data['percentages']:
                avg = np.mean(data['percentages'])
                std = np.std(data['percentages'])
                print(f"  {model}: {avg:.2f}% ± {std:.2f}% (n={len(data['percentages'])} exams)")
            else:
                print(f"  {model}: No results found")
        
        # Create plot
        print("\nGenerating plot...")
        plot_path = self.create_plot(model_results)
        
        # Save metadata
        print("\nSaving metadata...")
        self.save_metadata()
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)


def main():
    """Main entry point."""
    # Get config path
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    
    # Run evaluation pipeline
    pipeline = EvaluationPipeline(str(config_path))
    pipeline.run()


if __name__ == "__main__":
    main()
