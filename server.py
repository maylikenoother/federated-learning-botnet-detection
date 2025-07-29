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
import numpy as np
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ZeroDayResultsTracker:
    def __init__(self, experiment_name=None):
        self.experiment_name = experiment_name or f"zero_day_federated_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_dir = f"results/{self.experiment_name}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize result storage
        self.training_history = []
        self.evaluation_history = []
        self.client_metrics = {}
        self.zero_day_metrics = {}
        
        # Create CSV files
        self.training_csv = os.path.join(self.results_dir, "training_history.csv")
        self.evaluation_csv = os.path.join(self.results_dir, "evaluation_history.csv")
        self.client_csv = os.path.join(self.results_dir, "client_metrics.csv")
        self.zero_day_csv = os.path.join(self.results_dir, "zero_day_metrics.csv")
        
        # Initialize CSV files with headers
        self._init_csv_files()
        
        logger.info(f"📊 Zero-day experiment results will be saved to: {self.results_dir}")
    
    def _init_csv_files(self):
        """Initialize CSV files with headers"""
        # Training history CSV
        with open(self.training_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['round', 'timestamp', 'train_loss', 'train_accuracy', 'num_clients', 'total_examples'])
        
        # Evaluation history CSV
        with open(self.evaluation_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['round', 'timestamp', 'accuracy', 'loss', 'num_clients', 'total_examples'])
        
        # Client metrics CSV
        with open(self.client_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['round', 'client_id', 'phase', 'accuracy', 'loss', 'num_examples', 'missing_attack', 'timestamp'])
        
        # Zero-day metrics CSV
        with open(self.zero_day_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'round', 'client_id', 'missing_attack', 'zero_day_accuracy', 
                'zero_day_fp_rate', 'zero_day_detection_rate', 'zero_day_samples', 'timestamp'
            ])
    
    def log_training_round(self, round_num, metrics, client_metrics_list=None):
        """Log training round results with enhanced zero-day tracking"""
        timestamp = datetime.now().isoformat()
        
        # Calculate weighted average training accuracy if available
        train_accuracy = 0.0
        if client_metrics_list:
            total_samples = sum(num_examples for num_examples, _ in client_metrics_list)
            weighted_acc = sum(num_examples * m.get('accuracy', 0.0) for num_examples, m in client_metrics_list)
            train_accuracy = weighted_acc / total_samples if total_samples > 0 else 0.0
        
        record = {
            'round': round_num,
            'timestamp': timestamp,
            'train_loss': metrics.get('train_loss', 0.0),
            'train_accuracy': train_accuracy,
            'num_clients': metrics.get('num_clients', 0),
            'total_examples': metrics.get('total_examples', 0)
        }
        
        self.training_history.append(record)
        
        # Save to CSV
        with open(self.training_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                record['round'], record['timestamp'], record['train_loss'], 
                record['train_accuracy'], record['num_clients'], record['total_examples']
            ])
        
        # Display results
        print(f"\n{'='*60}")
        print(f"🎯 TRAINING ROUND {round_num} RESULTS")
        print(f"{'='*60}")
        print(f"📊 Train Loss: {record['train_loss']:.6f}")
        print(f"🎯 Train Accuracy: {record['train_accuracy']:.4f} ({record['train_accuracy']*100:.2f}%)")
        print(f"👥 Participating Clients: {record['num_clients']}")
        print(f"📈 Total Training Examples: {record['total_examples']}")
        print(f"🕒 Timestamp: {timestamp}")
        print(f"{'='*60}\n")
    
    def log_evaluation_round(self, round_num, metrics, client_metrics_list=None):
        """Log evaluation round results with zero-day analysis"""
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
            writer.writerow([
                record['round'], record['timestamp'], record['accuracy'], 
                record['loss'], record['num_clients'], record['total_examples']
            ])
        
        # Analyze zero-day performance
        zero_day_summary = self._analyze_zero_day_performance(client_metrics_list, round_num)
        
        # Display results
        print(f"\n{'='*60}")
        print(f"📊 EVALUATION ROUND {round_num} RESULTS")
        print(f"{'='*60}")
        print(f"🎯 Global Accuracy: {record['accuracy']:.4f} ({record['accuracy']*100:.2f}%)")
        print(f"📉 Global Loss: {record['loss']:.6f}")
        print(f"👥 Participating Clients: {record['num_clients']}")
        print(f"📈 Total Test Examples: {record['total_examples']}")
        
        if zero_day_summary:
            print(f"\n🚨 ZERO-DAY ATTACK DETECTION ANALYSIS:")
            print(f"   Average Zero-Day Accuracy: {zero_day_summary['avg_zero_day_accuracy']:.4f}")
            print(f"   Average False Positive Rate: {zero_day_summary['avg_fp_rate']:.4f}")
            print(f"   Clients with Zero-Day Data: {zero_day_summary['clients_with_zero_day']}")
            print(f"   Total Zero-Day Samples: {zero_day_summary['total_zero_day_samples']}")
        
        print(f"🕒 Timestamp: {timestamp}")
        print(f"{'='*60}\n")
    
    def log_client_metrics(self, round_num, client_metrics_list, phase):
        """Log individual client metrics with zero-day tracking"""
        timestamp = datetime.now().isoformat()
        
        print(f"\n📋 CLIENT METRICS - Round {round_num} ({phase.upper()})")
        print("-" * 50)
        
        with open(self.client_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            
            for i, (num_examples, metrics) in enumerate(client_metrics_list):
                accuracy = metrics.get('accuracy', 0.0)
                loss = metrics.get('loss', 0.0)
                missing_attack = metrics.get('missing_attack', 'Unknown')
                client_id = metrics.get('client_id', i)
                
                writer.writerow([
                    round_num, client_id, phase, accuracy, loss, 
                    num_examples, missing_attack, timestamp
                ])
                
                print(f"👤 Client {client_id}: Acc={accuracy:.4f}, Loss={loss:.6f}, "
                      f"Examples={num_examples}, Missing={missing_attack}")
                
                # Log zero-day specific metrics if available
                if phase == 'evaluation' and 'zero_day_accuracy' in metrics:
                    zero_day_acc = metrics.get('zero_day_accuracy', 0.0)
                    zero_day_fp = metrics.get('zero_day_fp_rate', 0.0)
                    zero_day_dr = metrics.get('zero_day_detection_rate', 0.0)
                    zero_day_samples = metrics.get('zero_day_samples', 0)
                    
                    print(f"   🚨 Zero-Day: Acc={zero_day_acc:.4f}, FP={zero_day_fp:.4f}, "
                          f"DR={zero_day_dr:.4f}, Samples={zero_day_samples}")
                    
                    # Save zero-day metrics
                    with open(self.zero_day_csv, 'a', newline='') as zd_f:
                        zd_writer = csv.writer(zd_f)
                        zd_writer.writerow([
                            round_num, client_id, missing_attack, zero_day_acc,
                            zero_day_fp, zero_day_dr, zero_day_samples, timestamp
                        ])
        
        print("-" * 50)
    
    def _analyze_zero_day_performance(self, client_metrics_list, round_num):
        """Analyze zero-day attack detection performance across clients"""
        if not client_metrics_list:
            return None
        
        zero_day_accuracies = []
        fp_rates = []
        total_zero_day_samples = 0
        clients_with_zero_day = 0
        
        for num_examples, metrics in client_metrics_list:
            if 'zero_day_accuracy' in metrics:
                zero_day_acc = metrics.get('zero_day_accuracy', 0.0)
                zero_day_fp = metrics.get('zero_day_fp_rate', 0.0)
                zero_day_samples = metrics.get('zero_day_samples', 0)
                
                if zero_day_samples > 0:
                    zero_day_accuracies.append(zero_day_acc)
                    fp_rates.append(zero_day_fp)
                    total_zero_day_samples += zero_day_samples
                    clients_with_zero_day += 1
        
        if zero_day_accuracies:
            return {
                'avg_zero_day_accuracy': np.mean(zero_day_accuracies),
                'avg_fp_rate': np.mean(fp_rates),
                'total_zero_day_samples': total_zero_day_samples,
                'clients_with_zero_day': clients_with_zero_day,
                'round': round_num
            }
        
        return None
    
    def save_final_summary(self):
        """Save comprehensive final experiment summary"""
        # Calculate summary statistics
        if self.evaluation_history:
            final_accuracy = self.evaluation_history[-1]['accuracy']
            final_loss = self.evaluation_history[-1]['loss']
            best_accuracy = max(r['accuracy'] for r in self.evaluation_history)
            best_round = max(range(len(self.evaluation_history)), 
                           key=lambda i: self.evaluation_history[i]['accuracy']) + 1
        else:
            final_accuracy = final_loss = best_accuracy = 0.0
            best_round = 0
        
        # Analyze zero-day performance over time
        zero_day_analysis = self._analyze_zero_day_trends()
        
        summary = {
            'experiment_name': self.experiment_name,
            'experiment_type': 'Zero-Day Federated Learning',
            'total_rounds': len(self.evaluation_history),
            'final_accuracy': final_accuracy,
            'final_loss': final_loss,
            'best_accuracy': best_accuracy,
            'best_round': best_round,
            'zero_day_analysis': zero_day_analysis,
            'training_history': self.training_history,
            'evaluation_history': self.evaluation_history,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save JSON summary
        with open(os.path.join(self.results_dir, 'experiment_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create comprehensive plots
        self._create_zero_day_plots()
        
        # Print final summary
        self._print_final_summary(summary)
        
        return summary
    
    def _analyze_zero_day_trends(self):
        """Analyze zero-day detection trends over rounds"""
        try:
            df = pd.read_csv(self.zero_day_csv)
            if len(df) == 0:
                return {}
            
            analysis = {
                'avg_zero_day_accuracy_per_round': {},
                'avg_fp_rate_per_round': {},
                'total_zero_day_samples_per_round': {},
                'clients_tested_per_round': {}
            }
            
            for round_num in df['round'].unique():
                round_data = df[df['round'] == round_num]
                
                # Filter out rows with zero samples
                valid_data = round_data[round_data['zero_day_samples'] > 0]
                
                if len(valid_data) > 0:
                    analysis['avg_zero_day_accuracy_per_round'][round_num] = valid_data['zero_day_accuracy'].mean()
                    analysis['avg_fp_rate_per_round'][round_num] = valid_data['zero_day_fp_rate'].mean()
                    analysis['total_zero_day_samples_per_round'][round_num] = valid_data['zero_day_samples'].sum()
                    analysis['clients_tested_per_round'][round_num] = len(valid_data)
            
            return analysis
        
        except Exception as e:
            logger.warning(f"Failed to analyze zero-day trends: {e}")
            return {}
    
    def _create_zero_day_plots(self):
        """Create comprehensive visualizations for zero-day federated learning"""
        if not self.evaluation_history:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Zero-Day Federated Learning Results - {self.experiment_name}', fontsize=16)
        
        rounds = [r['round'] for r in self.evaluation_history]
        accuracies = [r['accuracy'] for r in self.evaluation_history]
        losses = [r['loss'] for r in self.evaluation_history]
        
        # 1. Global Accuracy over rounds
        axes[0, 0].plot(rounds, accuracies, 'b-o', linewidth=2, markersize=6)
        axes[0, 0].set_title('Global Model Accuracy')
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 1)
        
        # 2. Global Loss over rounds
        axes[0, 1].plot(rounds, losses, 'r-o', linewidth=2, markersize=6)
        axes[0, 1].set_title('Global Model Loss')
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Training vs Evaluation Accuracy
        if self.training_history:
            train_rounds = [r['round'] for r in self.training_history]
            train_accuracies = [r['train_accuracy'] for r in self.training_history]
            axes[0, 2].plot(train_rounds, train_accuracies, 'g-o', label='Training', linewidth=2)
            axes[0, 2].plot(rounds, accuracies, 'b-o', label='Evaluation', linewidth=2)
            axes[0, 2].set_title('Training vs Evaluation Accuracy')
            axes[0, 2].set_xlabel('Round')
            axes[0, 2].set_ylabel('Accuracy')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Zero-Day Detection Performance
        try:
            zero_day_df = pd.read_csv(self.zero_day_csv)
            if len(zero_day_df) > 0:
                # Group by round and calculate average zero-day accuracy
                zd_by_round = zero_day_df[zero_day_df['zero_day_samples'] > 0].groupby('round').agg({
                    'zero_day_accuracy': 'mean',
                    'zero_day_fp_rate': 'mean'
                }).reset_index()
                
                if len(zd_by_round) > 0:
                    axes[1, 0].plot(zd_by_round['round'], zd_by_round['zero_day_accuracy'], 
                                  'purple', marker='s', linewidth=2, markersize=6)
                    axes[1, 0].set_title('Zero-Day Attack Detection Accuracy')
                    axes[1, 0].set_xlabel('Round')
                    axes[1, 0].set_ylabel('Zero-Day Detection Accuracy')
                    axes[1, 0].grid(True, alpha=0.3)
                    axes[1, 0].set_ylim(0, 1)
                    
                    # 5. False Positive Rate
                    axes[1, 1].plot(zd_by_round['round'], zd_by_round['zero_day_fp_rate'], 
                                  'orange', marker='d', linewidth=2, markersize=6)
                    axes[1, 1].set_title('Zero-Day False Positive Rate')
                    axes[1, 1].set_xlabel('Round')
                    axes[1, 1].set_ylabel('False Positive Rate')
                    axes[1, 1].grid(True, alpha=0.3)
        
        except Exception as e:
            logger.warning(f"Could not create zero-day plots: {e}")
        
        # 6. Client Performance Distribution
        try:
            client_df = pd.read_csv(self.client_csv)
            eval_df = client_df[client_df['phase'] == 'evaluation']
            if len(eval_df) > 0:
                latest_round = eval_df['round'].max()
                latest_data = eval_df[eval_df['round'] == latest_round]
                
                axes[1, 2].bar(range(len(latest_data)), latest_data['accuracy'], 
                             alpha=0.7, color='skyblue', edgecolor='navy')
                axes[1, 2].set_title(f'Client Accuracy Distribution (Round {latest_round})')
                axes[1, 2].set_xlabel('Client ID')
                axes[1, 2].set_ylabel('Accuracy')
                axes[1, 2].set_xticks(range(len(latest_data)))
                axes[1, 2].set_xticklabels([f"C{int(cid)}" for cid in latest_data['client_id']])
                axes[1, 2].grid(True, alpha=0.3, axis='y')
        
        except Exception as e:
            logger.warning(f"Could not create client distribution plot: {e}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'zero_day_federated_plots.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📈 Plots saved to: {self.results_dir}")
    
    def _print_final_summary(self, summary):
        """Print comprehensive final summary for zero-day federated learning"""
        print(f"\n{'='*80}")
        print(f"🎉 ZERO-DAY FEDERATED LEARNING EXPERIMENT COMPLETED")
        print(f"{'='*80}")
        print(f"📋 Experiment: {summary['experiment_name']}")
        print(f"🔬 Type: {summary['experiment_type']}")
        print(f"🔄 Total Rounds: {summary['total_rounds']}")
        print(f"🎯 Final Global Accuracy: {summary['final_accuracy']:.4f} ({summary['final_accuracy']*100:.2f}%)")
        print(f"📉 Final Global Loss: {summary['final_loss']:.6f}")
        print(f"🏆 Best Global Accuracy: {summary['best_accuracy']:.4f} ({summary['best_accuracy']*100:.2f}%) at Round {summary['best_round']}")
        
        # Zero-day specific summary
        zero_day_analysis = summary.get('zero_day_analysis', {})
        if zero_day_analysis:
            print(f"\n🚨 ZERO-DAY ATTACK DETECTION SUMMARY:")
            if 'avg_zero_day_accuracy_per_round' in zero_day_analysis:
                final_round_zd = max(zero_day_analysis['avg_zero_day_accuracy_per_round'].keys()) if zero_day_analysis['avg_zero_day_accuracy_per_round'] else 0
                if final_round_zd > 0:
                    final_zd_acc = zero_day_analysis['avg_zero_day_accuracy_per_round'][final_round_zd]
                    final_fp_rate = zero_day_analysis['avg_fp_rate_per_round'].get(final_round_zd, 0)
                    print(f"   Final Zero-Day Detection Accuracy: {final_zd_acc:.4f} ({final_zd_acc*100:.2f}%)")
                    print(f"   Final False Positive Rate: {final_fp_rate:.4f} ({final_fp_rate*100:.2f}%)")
        
        print(f"📁 Results saved to: {self.results_dir}")
        print(f"{'='*80}\n")

# Global results tracker
results_tracker = None

def fit_config(server_round: int) -> Dict[str, Union[bool, bytes, float, int, str]]:
    """Return training configuration dict for each round."""
    config = {
        "server_round": server_round,
        "local_epochs": 2,  # Increased for better learning
        "batch_size": 32,
        "learning_rate": 0.0005,  # Lower learning rate for stability
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
    accuracies = [num_examples * m.get("accuracy", 0.0) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    # Calculate weighted averages
    total_examples = sum(examples)
    if total_examples > 0:
        avg_loss = sum(losses) / total_examples
        avg_accuracy = sum(accuracies) / total_examples
    else:
        avg_loss = 0.0
        avg_accuracy = 0.0
    
    aggregated_metrics = {
        "train_loss": avg_loss,
        "train_accuracy": avg_accuracy,
        "num_clients": len(metrics),
        "total_examples": total_examples
    }
    
    return aggregated_metrics

def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate evaluation metrics from multiple clients with zero-day analysis."""
    global results_tracker
    
    # Extract metrics
    accuracies = [num_examples * m.get("accuracy", 0.0) for num_examples, m in metrics]
    losses = [num_examples * m.get("loss", 0.0) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    # Calculate weighted averages
    total_examples = sum(examples)
    if total_examples > 0:
        avg_accuracy = sum(accuracies) / total_examples
        avg_loss = sum(losses) / total_examples
    else:
        avg_accuracy = 0.0
        avg_loss = 0.0
    
    # Aggregate zero-day specific metrics
    zero_day_accuracies = []
    zero_day_fp_rates = []
    total_zero_day_samples = 0
    
    for num_examples, m in metrics:
        if 'zero_day_accuracy' in m and m.get('zero_day_samples', 0) > 0:
            zero_day_accuracies.append(m['zero_day_accuracy'])
            zero_day_fp_rates.append(m.get('zero_day_fp_rate', 0.0))
            total_zero_day_samples += m.get('zero_day_samples', 0)
    
    aggregated_metrics = {
        "accuracy": avg_accuracy,
        "loss": avg_loss,
        "num_clients": len(metrics),
        "total_examples": total_examples
    }
    
    # Add zero-day aggregated metrics
    if zero_day_accuracies:
        aggregated_metrics["avg_zero_day_accuracy"] = np.mean(zero_day_accuracies)
        aggregated_metrics["avg_zero_day_fp_rate"] = np.mean(zero_day_fp_rates)
        aggregated_metrics["total_zero_day_samples"] = total_zero_day_samples
        aggregated_metrics["clients_with_zero_day"] = len(zero_day_accuracies)
    
    return aggregated_metrics

class ZeroDayFedAvg(fl.server.strategy.FedAvg):
    """Custom FedAvg strategy with zero-day attack detection tracking"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_round = 0
    
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate training results and log zero-day metrics."""
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
            client_metrics = [(r.num_examples, r.metrics) for _, r in results] if results else None
            results_tracker.log_training_round(server_round, aggregated_metrics, client_metrics)
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation results and log zero-day detection metrics."""
        global results_tracker
        
        # Log client evaluation metrics (including zero-day)
        if results_tracker and results:
            client_metrics = [(r.num_examples, r.metrics) for _, r in results]
            results_tracker.log_client_metrics(server_round, client_metrics, "evaluation")
        
        # Call parent aggregation
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        
        # Log aggregated evaluation metrics
        if results_tracker and aggregated_metrics:
            client_metrics = [(r.num_examples, r.metrics) for _, r in results] if results else None
            results_tracker.log_evaluation_round(server_round, aggregated_metrics, client_metrics)
        
        return aggregated_loss, aggregated_metrics

def main():
    """Start the Zero-Day Federated Learning Server."""
    global results_tracker
    
    logger.info("🌸 Starting Zero-Day Federated Learning Server...")
    
    # Initialize results tracker
    results_tracker = ZeroDayResultsTracker()
    
    # Create strategy with zero-day tracking
    strategy = ZeroDayFedAvg(
        fraction_fit=0.8,  # 80% of available clients for training
        fraction_evaluate=1.0,  # All clients for evaluation to test zero-day detection
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
        logger.info("🚀 Starting server on 0.0.0.0:8080")
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=10),  # More rounds for better learning
            strategy=strategy,
        )
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server failed: {e}")
    finally:
        # Save final results
        if results_tracker:
            results_tracker.save_final_summary()
            logger.info("📊 Final zero-day experiment results saved successfully!")

if __name__ == "__main__":
    main()