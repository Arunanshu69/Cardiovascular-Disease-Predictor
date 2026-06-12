"""
Ablation Study Script
Runs 4 experiments to compare different model configurations:
1. XGBoost only (Baseline)
2. Diffusion + XGBoost (Effect of diffusion)
3. GA-PSO + XGBoost (Effect of optimization)
4. Diffusion + GA-PSO + XGBoost (Full proposed model)
"""

import subprocess
import pandas as pd
import os
import time
from datetime import datetime

def run_experiment(name, use_optimization, use_diffusion, data_path, target_column, output_dir):
    """Run a single experiment with given configuration."""
    print(f"\n{'='*60}")
    print(f"Running Experiment: {name}")
    print(f"{'='*60}")
    
    # Build command
    cmd = [
        "python", "main.py",
        "--data_path", data_path,
        "--target_column", target_column,
        "--output_dir", output_dir
    ]
    
    if use_optimization:
        cmd.append("--use_optimization")
    
    if use_diffusion:
        cmd.append("--use_diffusion")
    
    # Run experiment
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    duration = end_time - start_time
    
    # Parse results from output
    output = result.stdout
    
    # Extract metrics
    metrics = {
        'experiment': name,
        'use_optimization': use_optimization,
        'use_diffusion': use_diffusion,
        'duration_seconds': duration,
        'success': result.returncode == 0
    }
    
    # Parse metrics from output
    lines = output.split('\n')
    for line in lines:
        if 'accuracy' in line.lower() and ':' in line:
            try:
                metrics['accuracy'] = float(line.split(':')[1].strip())
            except:
                pass
        elif 'precision' in line.lower() and ':' in line:
            try:
                metrics['precision'] = float(line.split(':')[1].strip())
            except:
                pass
        elif 'recall' in line.lower() and ':' in line:
            try:
                metrics['recall'] = float(line.split(':')[1].strip())
            except:
                pass
        elif 'f1_score' in line.lower() and ':' in line:
            try:
                metrics['f1_score'] = float(line.split(':')[1].strip())
            except:
                pass
        elif 'roc_auc' in line.lower() and ':' in line:
            try:
                metrics['roc_auc'] = float(line.split(':')[1].strip())
            except:
                pass
        elif 'CV AUC' in line:
            try:
                metrics['cv_auc'] = float(line.split(':')[1].strip().split()[0])
            except:
                pass
    
    print(f"Experiment completed in {duration:.2f} seconds")
    print(f"Success: {metrics['success']}")
    
    if 'accuracy' in metrics:
        print(f"Accuracy: {metrics['accuracy']:.4f}")
    if 'recall' in metrics:
        print(f"Recall: {metrics['recall']:.4f}")
    if 'f1_score' in metrics:
        print(f"F1-Score: {metrics['f1_score']:.4f}")
    if 'roc_auc' in metrics:
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    
    return metrics

def main():
    """Run all ablation experiments."""
    # Configuration
    data_path = "data/framingham.csv"
    target_column = "TenYearCHD"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define experiments with separate output directories
    experiments = [
        {
            'name': 'XGBoost Only (Baseline)',
            'use_optimization': False,
            'use_diffusion': False,
            'output_dir': f"results_ablation/1_baseline_{timestamp}"
        },
        {
            'name': 'Diffusion + XGBoost',
            'use_optimization': False,
            'use_diffusion': True,
            'output_dir': f"results_ablation/2_diffusion_{timestamp}"
        },
        {
            'name': 'GA-PSO + XGBoost',
            'use_optimization': True,
            'use_diffusion': False,
            'output_dir': f"results_ablation/3_optimization_{timestamp}"
        },
        {
            'name': 'Diffusion + GA-PSO + XGBoost (Full Model)',
            'use_optimization': True,
            'use_diffusion': True,
            'output_dir': f"results_ablation/4_full_model_{timestamp}"
        }
    ]
    
    # Create ablation results directory
    os.makedirs("results_ablation", exist_ok=True)
    
    # Run experiments
    results = []
    
    for exp in experiments:
        # Create output directory for this experiment
        os.makedirs(exp['output_dir'], exist_ok=True)
        
        metrics = run_experiment(
            exp['name'],
            exp['use_optimization'],
            exp['use_diffusion'],
            data_path,
            target_column,
            exp['output_dir']
        )
        metrics['output_dir'] = exp['output_dir']
        results.append(metrics)
        
        # Save results after each experiment
        df = pd.DataFrame(results)
        output_file = f"results_ablation/ablation_results_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("ABLATION STUDY SUMMARY")
    print(f"{'='*60}")
    
    df = pd.DataFrame(results)
    
    # Display key metrics
    key_metrics = ['experiment', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'cv_auc', 'duration_seconds']
    print(df[key_metrics].to_string(index=False))
    
    # Calculate improvements
    if len(results) == 4:
        baseline = results[0]
        full_model = results[3]
        
        print(f"\n{'='*60}")
        print("IMPROVEMENTS (Full Model vs Baseline)")
        print(f"{'='*60}")
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            if metric in baseline and metric in full_model:
                improvement = (full_model[metric] - baseline[metric]) / baseline[metric] * 100
                print(f"{metric}: {improvement:+.2f}%")
    
    print(f"\n{'='*60}")
    print("ABLATION STUDY COMPLETED")
    print(f"{'='*60}")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()
