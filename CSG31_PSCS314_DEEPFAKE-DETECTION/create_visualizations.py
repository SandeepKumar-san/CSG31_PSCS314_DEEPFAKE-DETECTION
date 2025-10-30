import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class VisualizationGenerator:
    def __init__(self):
        self.load_results()
        
    def load_results(self):
        """Load experiment results"""
        try:
            with open('experiment_results.json', 'r') as f:
                self.results = json.load(f)
            print("Loaded experiment results successfully")
        except FileNotFoundError:
            print("No experiment results found. Run run_experiments.py first.")
            return
    
    def create_training_plots(self):
        """Create training progress visualizations"""
        history = self.results['training_history']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Training Loss
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Validation Loss
        ax2.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax2.set_title('Validation Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Training Accuracy
        ax3.plot(epochs, history['train_acc'], 'g-', label='Training Accuracy', linewidth=2)
        ax3.set_title('Training Accuracy', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Validation Accuracy
        ax4.plot(epochs, history['val_acc'], 'orange', label='Validation Accuracy', linewidth=2)
        ax4.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Combined plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax2 = ax.twinx()
        
        # Loss on left y-axis
        line1 = ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        line2 = ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        
        # Accuracy on right y-axis
        line3 = ax2.plot(epochs, history['train_acc'], 'g--', label='Training Accuracy', linewidth=2)
        line4 = ax2.plot(epochs, history['val_acc'], 'orange', linestyle='--', label='Validation Accuracy', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Training Progress: Loss and Accuracy', fontsize=14, fontweight='bold')
        
        # Combine legends
        lines = line1 + line2 + line3 + line4
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='center right')
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('combined_training_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_confusion_matrix(self):
        """Create confusion matrix visualization"""
        cm = np.array(self.results['evaluation']['confusion_matrix'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Real', 'Fake'], 
                   yticklabels=['Real', 'Fake'])
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_roc_curve(self):
        """Create ROC curve visualization"""
        fpr = self.results['evaluation']['fpr']
        tpr = self.results['evaluation']['tpr']
        roc_auc = self.results['evaluation']['roc_auc']
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_ablation_plots(self):
        """Create ablation study visualizations"""
        ablation = self.results['ablation_study']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Sequence Length
        seq_data = ablation['sequence_length']
        seq_names = list(seq_data.keys())
        seq_acc = [seq_data[k]['accuracy'] for k in seq_names]
        
        axes[0,0].bar(seq_names, seq_acc, color='skyblue', edgecolor='navy')
        axes[0,0].set_title('Sequence Length Ablation', fontweight='bold')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # LSTM Hidden Units
        lstm_data = ablation['lstm_hidden_units']
        lstm_names = list(lstm_data.keys())
        lstm_acc = [lstm_data[k]['accuracy'] for k in lstm_names]
        
        axes[0,1].bar(lstm_names, lstm_acc, color='lightgreen', edgecolor='darkgreen')
        axes[0,1].set_title('LSTM Hidden Units Ablation', fontweight='bold')
        axes[0,1].set_ylabel('Accuracy')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Feature Extractor
        feat_data = ablation['feature_extractor']
        feat_names = list(feat_data.keys())
        feat_acc = [feat_data[k]['accuracy'] for k in feat_names]
        
        axes[1,0].bar(feat_names, feat_acc, color='lightcoral', edgecolor='darkred')
        axes[1,0].set_title('Feature Extractor Ablation', fontweight='bold')
        axes[1,0].set_ylabel('Accuracy')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Face Alignment
        align_data = ablation['face_alignment']
        align_names = list(align_data.keys())
        align_acc = [align_data[k]['accuracy'] for k in align_names]
        
        axes[1,1].bar(align_names, align_acc, color='gold', edgecolor='orange')
        axes[1,1].set_title('Face Alignment Impact', fontweight='bold')
        axes[1,1].set_ylabel('Accuracy')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('ablation_study.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_comparison_plot(self):
        """Create method comparison visualization"""
        comp = self.results['comparison_study']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Performance Comparison
        methods = comp['methods']
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        x = np.arange(len(methods))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = comp[metric]
            ax1.bar(x + i*width, values, width, label=metric.replace('_', ' ').title())
        
        ax1.set_xlabel('Methods')
        ax1.set_ylabel('Score')
        ax1.set_title('Performance Comparison Across Methods', fontweight='bold')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(methods, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Processing Time Comparison
        processing_times = comp['processing_time']
        colors = ['red' if method == 'Our CNN-LSTM' else 'lightblue' for method in methods]
        
        ax2.bar(methods, processing_times, color=colors, edgecolor='navy')
        ax2.set_xlabel('Methods')
        ax2.set_ylabel('Processing Time (seconds)')
        ax2.set_title('Processing Time Comparison', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('method_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_performance_analysis(self):
        """Create performance analysis visualizations"""
        perf = self.results['performance_analysis']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Processing Time Breakdown
        times = perf['processing_times']
        components = list(times.keys())[:-1]  # Exclude 'total'
        time_values = [times[comp] for comp in components]
        
        ax1.pie(time_values, labels=components, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Processing Time Breakdown', fontweight='bold')
        
        # Memory Usage
        memory = perf['memory_usage']
        mem_types = list(memory.keys())
        mem_values = list(memory.values())
        
        ax2.bar(mem_types, mem_values, color=['lightblue', 'lightgreen', 'lightcoral'])
        ax2.set_title('Memory Usage Analysis', fontweight='bold')
        ax2.set_ylabel('Memory (MB/GB)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Scalability - Throughput
        batch_sizes = perf['scalability']['batch_sizes']
        throughput = perf['scalability']['throughput']
        
        ax3.plot(batch_sizes, throughput, 'bo-', linewidth=2, markersize=8)
        ax3.set_title('Throughput vs Batch Size', fontweight='bold')
        ax3.set_xlabel('Batch Size')
        ax3.set_ylabel('Throughput (videos/sec)')
        ax3.grid(True, alpha=0.3)
        
        # Scalability - Memory Usage
        memory_usage_scale = perf['scalability']['memory_usage']
        
        ax4.plot(batch_sizes, memory_usage_scale, 'ro-', linewidth=2, markersize=8)
        ax4.set_title('Memory Usage vs Batch Size', fontweight='bold')
        ax4.set_xlabel('Batch Size')
        ax4.set_ylabel('GPU Memory (GB)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_results_table(self):
        """Create results summary table"""
        eval_results = self.results['evaluation']
        
        # Create summary table
        summary_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'],
            'Value': [
                f"{eval_results['accuracy']:.3f}",
                f"{eval_results['precision']:.3f}",
                f"{eval_results['recall']:.3f}",
                f"{eval_results['f1_score']:.3f}",
                f"{eval_results['roc_auc']:.3f}"
            ]
        }
        
        df = pd.DataFrame(summary_data)
        
        # Create table visualization
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('Model Performance Summary', fontsize=14, fontweight='bold', pad=20)
        plt.savefig('results_table.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_sample_predictions(self):
        """Create sample prediction visualization"""
        y_true = self.results['evaluation']['y_true'][:20]  # First 20 samples
        y_pred_proba = self.results['evaluation']['y_pred_proba'][:20]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = range(len(y_true))
        colors = ['green' if true == (prob > 0.5) else 'red' 
                 for true, prob in zip(y_true, y_pred_proba)]
        
        bars = ax.bar(x, y_pred_proba, color=colors, alpha=0.7, edgecolor='black')
        
        # Add threshold line
        ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold')
        
        # Add true labels as text
        for i, (true_label, prob) in enumerate(zip(y_true, y_pred_proba)):
            ax.text(i, prob + 0.05, f'T:{int(true_label)}', ha='center', fontweight='bold')
        
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Prediction Probability')
        ax.set_title('Sample Predictions (Green=Correct, Red=Incorrect)', fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sample_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print("Generating visualizations...")
        
        self.create_training_plots()
        print("✓ Training progress plots created")
        
        self.create_confusion_matrix()
        print("✓ Confusion matrix created")
        
        self.create_roc_curve()
        print("✓ ROC curve created")
        
        self.create_ablation_plots()
        print("✓ Ablation study plots created")
        
        self.create_comparison_plot()
        print("✓ Method comparison plots created")
        
        self.create_performance_analysis()
        print("✓ Performance analysis plots created")
        
        self.create_results_table()
        print("✓ Results summary table created")
        
        self.create_sample_predictions()
        print("✓ Sample predictions plot created")
        
        print("\nAll visualizations saved as PNG files!")
        print("Files created:")
        print("- training_progress.png")
        print("- combined_training_progress.png")
        print("- confusion_matrix.png")
        print("- roc_curve.png")
        print("- ablation_study.png")
        print("- method_comparison.png")
        print("- performance_analysis.png")
        print("- results_table.png")
        print("- sample_predictions.png")

def main():
    generator = VisualizationGenerator()
    generator.generate_all_visualizations()

if __name__ == "__main__":
    main()