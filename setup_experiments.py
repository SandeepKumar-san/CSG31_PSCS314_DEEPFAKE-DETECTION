#!/usr/bin/env python3
"""
Quick setup script to run all experiments and generate results
"""

import subprocess
import sys
import os

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:")
        print(e.stdout)
        print(e.stderr)
        return False
    except FileNotFoundError:
        print(f"Script {script_name} not found!")
        return False

def install_requirements():
    """Install required packages"""
    requirements = [
        'torch', 'torchvision', 'numpy', 'matplotlib', 
        'seaborn', 'pandas', 'scikit-learn', 'opencv-python',
        'mtcnn', 'streamlit', 'plotly', 'pillow'
    ]
    
    print("Installing required packages...")
    for package in requirements:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                         check=True, capture_output=True)
            print(f"✓ {package}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")

def main():
    print("🚀 Setting up deepfake detection experiments...")
    print("This will generate all results, visualizations, and analysis files.")
    
    # Check if we're in the right directory
    if not os.path.exists('dashboard.py'):
        print("⚠️  Warning: dashboard.py not found. Make sure you're in the project directory.")
    
    # Install requirements
    install_requirements()
    
    # Run experiments in sequence
    scripts = [
        ('run_experiments.py', 'Collecting quantitative results and performance metrics'),
        ('create_visualizations.py', 'Generating plots and figures'),
        ('analyze_results.py', 'Creating analysis and LaTeX tables')
    ]
    
    success_count = 0
    for script, description in scripts:
        if run_script(script, description):
            success_count += 1
        else:
            print(f"❌ Failed to run {script}")
            break
    
    print(f"\n{'='*50}")
    print("📊 EXPERIMENT SETUP COMPLETE")
    print(f"{'='*50}")
    
    if success_count == len(scripts):
        print("✅ All experiments completed successfully!")
        print("\n📁 Generated files:")
        
        files_to_check = [
            'experiment_results.json',
            'training_progress.png',
            'confusion_matrix.png',
            'roc_curve.png',
            'ablation_study.png',
            'method_comparison.png',
            'performance_analysis.png',
            'results_table.png',
            'sample_predictions.png',
            'latex_tables.tex',
            'results_summary.md',
            'paper_experimental_results.tex',
            'statistical_analysis.md'
        ]
        
        for file in files_to_check:
            if os.path.exists(file):
                print(f"   ✓ {file}")
            else:
                print(f"   ✗ {file} (missing)")
        
        print("\n🎯 Next steps:")
        print("1. Review the generated visualizations (PNG files)")
        print("2. Copy LaTeX tables from latex_tables.tex to your paper")
        print("3. Include the experimental results section in your paper")
        print("4. Add the generated figures to your paper")
        print("5. Review results_summary.md for key findings")
        
        print("\n📝 For your research paper:")
        print("- Use paper_experimental_results.tex as your Results section")
        print("- Include the PNG figures in your paper")
        print("- Copy tables from latex_tables.tex")
        print("- Reference the statistical analysis for significance testing")
        
    else:
        print("❌ Some experiments failed. Check the error messages above.")
        print("💡 Try running the scripts individually to debug issues.")

if __name__ == "__main__":
    main()