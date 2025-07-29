import flwr as fl
from flwr.common import Metrics
from typing import List, Tuple, Dict, Optional, Union
import logging
import json
import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResultsTracker:
    def __init__(self, experiment_name=None):
        self.experiment_name = experiment_name or f"federated_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_dir = f"results/{self.experiment_name}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize result storage
        self.training_history = []
        self.evaluation_history = []
        self.client_metrics = {}
        
        # Create CSV files
        self.training_csv = os.path.join(self.results_dir, "training_history.csv")
        self.evaluation_csv = os.path.join(self.results_dir, "evaluation_history.csv")
        self.client_csv = os.path.join(self.results_dir, "client_metrics.csv")
        
        # Initialize CSV files with headers
        self._init_csv_files()
        
        logger.info(f"📊 Results will be saved to: {self.results_dir}")
    
    def _init_csv_files(self):
        """Initialize CSV files with headers"""
        # Training history CSV
        with open(self.training_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['round', 'timestamp', 'train_loss', 'num_clients', 'total_examples'])
        
        # Evaluation history CSV
        with open(self.evaluation_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['round', 'timestamp', 'accuracy', 'loss', 'num_clients', 'total_examples'])
        
        # Client metrics CSV
        with open(self.client_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['round', 'client_id', 'phase', 'accuracy', 'loss', 'num_examples', 'timestamp'])
    
    def log_training_round(self, round_num, metrics):
        """Log training round results"""
        timestamp = datetime.now().isoformat()
        
        record = {
            'round': round_num,
            'timestamp': timestamp,
            'train_loss': metrics.get('train_loss', 0.0),
            'num_clients': metrics.get('num_clients', 0),
            'total_examples': metrics.get('total_examples', 0)
        }
        
        self.training_history.append(record)
        
        # Save to CSV
        with open(self.training_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([record['round'], record['timestamp'], record['train_loss'], 
                           record['num_clients'], record['total_examples']])
        
        # Display results
        print(f"\n{'='*60}")
        print(f"🎯 TRAINING ROUND {round_num} RESULTS")
        print(f"{'='*60}")
        print(f"📊 Train Loss: {record['train_loss']:.6f}")
        print(f"👥 Participating Clients: {record['num_clients']}")
        print(f"📈 Total Training Examples: {record['total_examples']}")
        print(f"🕒 Timestamp: {timestamp}")
        print(f"{'='*60}\n")
    
    def log_evaluation_round(self, round_num, metrics):
        """Log evaluation round results"""
        timestamp = datetime.now().isoformat()
        
        record = {
            'round': round_num,
            'timestamp': timestamp,
            'accuracy': metrics.get('accuracy', 0.0),
            'loss': metrics.get('loss', 0.0),
            'num_clients': metrics.get('num_clients', 0),
            'total_examples': metrics.get('total_examples', 0)
        }
        
        self.evaluation_history.append(record)
        
        # Save to CSV
        with open(self.evaluation_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([record['round'], record['timestamp'], record['accuracy'], 
                           record['loss'], record['num_clients'], record['total_examples']])
        
        # Display results
        print(f"\n{'='*60}")
        print(f"📊 EVALUATION ROUND {round_num} RESULTS")
        print(f"{'='*60}")
        print(f"🎯 Global Accuracy: {record['accuracy']:.4f} ({record['accuracy']*100:.2f}%)")
        print(f"📉 Global Loss: {record['loss']:.6f}")
        print(f"👥 Participating Clients: {record['num_clients']}")
        print(f"📈 Total Test Examples: {record['total_examples']}")
        print(f"🕒 Timestamp: {timestamp}")
        print(f"{'='*60}\n")
    
    def log_client_metrics(self, round_num, client_metrics_list, phase):
        """Log individual client metrics"""
        timestamp = datetime.now().isoformat()
        
        print(f"\n📋 CLIENT METRICS - Round {round_num} ({phase.upper()})")
        print("-" * 50)
        
        with open(self.client_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            
            for i, (num_examples, metrics) in enumerate(client_metrics_list):
                accuracy = metrics.get('accuracy', 0.0)
                loss = metrics.get('loss', 0.0)
                
                writer.writerow([round_num, f'client_{i}', phase, accuracy, loss, num_examples, timestamp])
                
                print(f"👤 Client {i}: Acc={accuracy:.4f}, Loss={loss:.6f}, Examples={num_examples}")
        
        print("-" * 50)
    
    def save_final_summary(self):
        """Save final experiment summary"""
        summary = {
            'experiment_name': self.experiment_name,
            'total_rounds': len(self.evaluation_history),
            'final_accuracy': self.evaluation_history[-1]['accuracy'] if self.evaluation_history else 0.0,
            'final_loss': self.evaluation_history[-1]['loss'] if self.evaluation_history else 0.0,
            'best_accuracy': max((r['accuracy'] for r in self.evaluation_history), default=0.0),
            'best_round': max(range(len(self.evaluation_history)), 
                            key=lambda i: self.evaluation_history[i]['accuracy'], default=0) + 1,
            'training_history': self.training_history,
            'evaluation_history': self.evaluation_history,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save JSON summary
        with open(os.path.join(self.results_dir, 'experiment_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create plots
        self._create_plots()
        
        # Print final summary
        self._print_final_summary(summary)
        
        return summary
    
    def _create_plots(self):
        """Create visualization plots"""
        if not self.evaluation_history:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Federated Learning Results - {self.experiment_name}', fontsize=16)
        
        rounds = [r['round'] for r in self.evaluation_history]
        accuracies = [r['accuracy'] for r in self.evaluation_history]
        losses = [r['loss'] for r in self.evaluation_history]
        
        # Accuracy over rounds
        axes[0, 0].plot(rounds, accuracies, 'b-o', linewidth=2, markersize=6)
        axes[0, 0].set_title('Global Accuracy Over Rounds')
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 1)
        
        # Loss over rounds
        axes[0, 1].plot(rounds, losses, 'r-o', linewidth=2, markersize=6)
        axes[0, 1].set_title('Global Loss Over Rounds')
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Training loss
        if self.training_history:
            train_rounds = [r['round'] for r in self.training_history]
            train_losses = [r['train_loss'] for r in self.training_history]
            axes[1, 0].plot(train_rounds, train_losses, 'g-o', linewidth=2, markersize=6)
            axes[1, 0].set_title('Training Loss Over Rounds')
            axes[1, 0].set_xlabel('Round')
            axes[1, 0].set_ylabel('Training Loss')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Accuracy improvement
        if len(accuracies) > 1:
            improvements = [accuracies[i] - accuracies[i-1] for i in range(1, len(accuracies))]
            axes[1, 1].bar(rounds[1:], improvements, alpha=0.7)
            axes[1, 1].set_title('Accuracy Improvement Per Round')
            axes[1, 1].set_xlabel('Round')
            axes[1, 1].set_ylabel('Accuracy Change')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'training_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Plots saved to: {os.path.join(self.results_dir, 'training_plots.png')}")
    
    def _print_final_summary(self, summary):
        """Print comprehensive final summary"""
        print(f"\n{'='*80}")
        print(f"🎉 FEDERATED LEARNING EXPERIMENT COMPLETED")
        print(f"{'='*80}")
        print(f"📋 Experiment: {summary['experiment_name']}")
        print(f"🔄 Total Rounds: {summary['total_rounds']}")
        print(f"🎯 Final Accuracy: {summary['final_accuracy']:.4f} ({summary['final_accuracy']*100:.2f}%)")
        print(f"📉 Final Loss: {summary['final_loss']:.6f}")
        print(f"🏆 Best Accuracy: {summary['best_accuracy']:.4f} ({summary['best_accuracy']*100:.2f}%) at Round {summary['best_round']}")
        print(f"📁 Results saved to: {self.results_dir}")
        print(f"{'='*80}\n")

# Global results tracker
results_tracker = None

def fit_config(server_round: int) -> Dict[str, Union[bool, bytes, float, int, str]]:
    """Return training configuration dict for each round."""
    config = {
        "server_round": server_round,
        "local_epochs": 1,
        "batch_size": 64,
        "learning_rate": 0.001,
    }
    return config

def evaluate_config(server_round: int) -> Dict[str, Union[bool, bytes, float, int, str]]:
    """Return evaluation configuration dict for each round."""
    config = {
        "server_round": server_round,
    }
    return config

def fit_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate fit metrics (training metrics) from multiple clients."""
    global results_tracker
    
    # Extract training losses and examples
    losses = [num_examples * m.get("loss", 0.0) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    # Calculate weighted average loss
    if sum(examples) > 0:
        avg_loss = sum(losses) / sum(examples)
    else:
        avg_loss = 0.0
    
    aggregated_metrics = {
        "train_loss": avg_loss,
        "num_clients": len(metrics),
        "total_examples": sum(examples)
    }
    
    return aggregated_metrics

def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate evaluation metrics from multiple clients."""
    global results_tracker
    
    # Extract metrics
    accuracies = [num_examples * m.get("accuracy", 0.0) for num_examples, m in metrics]
    losses = [num_examples * m.get("loss", 0.0) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    # Calculate weighted averages
    if sum(examples) > 0:
        avg_accuracy = sum(accuracies) / sum(examples)
        avg_loss = sum(losses) / sum(examples)
    else:
        avg_accuracy = 0.0
        avg_loss = 0.0
    
    aggregated_metrics = {
        "accuracy": avg_accuracy,
        "loss": avg_loss,
        "num_clients": len(metrics),
        "total_examples": sum(examples)
    }
    
    return aggregated_metrics

class CustomStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_round = 0
    
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate training results and log them."""
        global results_tracker
        
        self.current_round = server_round
        
        # Log client training metrics
        if results_tracker and results:
            client_metrics = [(r.num_examples, r.metrics) for _, r in results]
            results_tracker.log_client_metrics(server_round, client_metrics, "training")
        
        # Call parent aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        # Log aggregated training metrics
        if results_tracker and aggregated_metrics:
            results_tracker.log_training_round(server_round, aggregated_metrics)
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation results and log them."""
        global results_tracker
        
        # Log client evaluation metrics
        if results_tracker and results:
            client_metrics = [(r.num_examples, r.metrics) for _, r in results]
            results_tracker.log_client_metrics(server_round, client_metrics, "evaluation")
        
        # Call parent aggregation
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        
        # Log aggregated evaluation metrics
        if results_tracker and aggregated_metrics:
            results_tracker.log_evaluation_round(server_round, aggregated_metrics)
        
        return aggregated_loss, aggregated_metrics

def main():
    """Start the Flower server with results tracking."""
    global results_tracker
    
    logger.info("🌸 Starting Flower Server with Results Tracking...")
    
    # Initialize results tracker
    results_tracker = ResultsTracker()
    
    # Create strategy with results tracking
    strategy = CustomStrategy(
        fraction_fit=0.8,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
    )
    
    try:
        # Start server
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=5),
            strategy=strategy,
        )
    except Exception as e:
        logger.error(f"Server failed: {e}")
    finally:
        # Save final results
        if results_tracker:
            results_tracker.save_final_summary()
            logger.info("📊 Final results saved successfully!")

if __name__ == "__main__":
    main()