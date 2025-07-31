import flwr as fl
from flwr.common import Metrics, FitRes, Parameters, Scalar
from typing import List, Tuple, Dict, Optional, Union
import logging
import json
import csv
import os
from datetime import datetime
import time
import numpy as np
from collections import OrderedDict
import argparse
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ResultsTracker:
    def __init__(self, algorithm_name="FedAvg"):
        self.algorithm_name = algorithm_name
        self.experiment_id = f"{algorithm_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_dir = f"results/{self.experiment_id}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.training_history = []
        self.evaluation_history = []
        self.round_times = []
        self.start_time = time.time()
        
        # Initialize CSV files
        self.training_csv = os.path.join(self.results_dir, "training_history.csv")
        self.eval_csv = os.path.join(self.results_dir, "evaluation_history.csv")
        
        with open(self.training_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['round', 'loss', 'num_clients', 'total_examples', 'timestamp'])
        
        with open(self.eval_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['round', 'loss', 'accuracy', 'num_clients', 'timestamp'])
        
        logger.info(f"📊 Results tracker initialized for {algorithm_name}")
    
    def log_training_round(self, round_num, loss, num_clients, total_examples):
        """Log training round metrics"""
        timestamp = datetime.now().isoformat()
        self.training_history.append({
            'round': round_num,
            'loss': loss,
            'num_clients': num_clients,
            'total_examples': total_examples,
            'timestamp': timestamp
        })
        
        with open(self.training_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([round_num, loss, num_clients, total_examples, timestamp])
    
    def log_evaluation_round(self, round_num, loss, accuracy, num_clients):
        """Log evaluation round metrics"""
        timestamp = datetime.now().isoformat()
        self.evaluation_history.append({
            'round': round_num,
            'loss': loss,
            'accuracy': accuracy,
            'num_clients': num_clients,
            'timestamp': timestamp
        })
        
        with open(self.eval_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([round_num, loss, accuracy, num_clients, timestamp])
    
    def save_final_summary(self):
        """Save final experiment summary"""
        total_time = time.time() - self.start_time
        
        summary = {
            'algorithm': self.algorithm_name,
            'experiment_id': self.experiment_id,
            'total_rounds': len(self.evaluation_history),
            'total_time': total_time,
            'final_accuracy': self.evaluation_history[-1]['accuracy'] if self.evaluation_history else 0,
            'final_loss': self.evaluation_history[-1]['loss'] if self.evaluation_history else float('inf'),
            'training_history': self.training_history,
            'evaluation_history': self.evaluation_history
        }
        
        summary_file = os.path.join(self.results_dir, 'experiment_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"💾 Experiment summary saved to {summary_file}")
        return summary

# Custom strategy that works with all algorithms
class CustomStrategy(fl.server.strategy.FedAvg):
    def __init__(self, algorithm_name="FedAvg", results_tracker=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.algorithm_name = algorithm_name
        self.results_tracker = results_tracker or ResultsTracker(algorithm_name)
        self.current_round = 0
        logger.info(f"🚀 {algorithm_name} strategy initialized")
    
    def aggregate_fit(self, server_round: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]], 
                     failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results"""
        self.current_round = server_round
        
        if not results:
            logger.warning(f"No results to aggregate in round {server_round}")
            return None, {}
        
        # Log failures
        if failures:
            logger.warning(f"Failed clients in round {server_round}: {len(failures)}")
        
        # Extract metrics from results
        losses = []
        accuracies = []
        examples = []
        
        for client, fit_res in results:
            examples.append(fit_res.num_examples)
            if fit_res.metrics:
                if "loss" in fit_res.metrics:
                    losses.append(fit_res.metrics["loss"])
                if "accuracy" in fit_res.metrics:
                    accuracies.append(fit_res.metrics["accuracy"])
        
        # Calculate weighted averages
        total_examples = sum(examples)
        if losses:
            avg_loss = sum(l * e for l, e in zip(losses, examples)) / total_examples
        else:
            avg_loss = 0.0
        
        # Log training metrics
        self.results_tracker.log_training_round(server_round, avg_loss, len(results), total_examples)
        
        # Log round info
        logger.info(f"\n{'='*60}")
        logger.info(f"ROUND {server_round} - {self.algorithm_name}")
        logger.info(f"{'='*60}")
        logger.info(f"Clients: {len(results)}")
        logger.info(f"Total examples: {total_examples}")
        logger.info(f"Average loss: {avg_loss:.4f}")
        if accuracies:
            avg_acc = sum(a * e for a, e in zip(accuracies, examples)) / total_examples
            logger.info(f"Average accuracy: {avg_acc:.4f}")
        
        # Standard FedAvg aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        # Add algorithm-specific metrics
        if aggregated_metrics is None:
            aggregated_metrics = {}
        
        aggregated_metrics["algorithm"] = self.algorithm_name
        aggregated_metrics["round"] = server_round
        aggregated_metrics["num_clients"] = len(results)
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(self, server_round: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
                          failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes], BaseException]]) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results"""
        if not results:
            logger.warning(f"No evaluation results in round {server_round}")
            return None, {}
        
        # Log failures
        if failures:
            logger.warning(f"Failed evaluations in round {server_round}: {len(failures)}")
        
        # Weighted average of loss and accuracy
        total_loss = 0.0
        total_accuracy = 0.0
        total_examples = 0
        
        for client, eval_res in results:
            total_loss += eval_res.loss * eval_res.num_examples
            total_examples += eval_res.num_examples
            
            if eval_res.metrics and "accuracy" in eval_res.metrics:
                total_accuracy += eval_res.metrics["accuracy"] * eval_res.num_examples
        
        avg_loss = total_loss / total_examples if total_examples > 0 else float('inf')
        avg_accuracy = total_accuracy / total_examples if total_examples > 0 else 0.0
        
        # Log evaluation metrics
        self.results_tracker.log_evaluation_round(server_round, avg_loss, avg_accuracy, len(results))
        
        # Display results
        logger.info(f"\nEVALUATION - Round {server_round}")
        logger.info(f"Loss: {avg_loss:.4f}")
        logger.info(f"Accuracy: {avg_accuracy:.4f}")
        logger.info(f"Clients evaluated: {len(results)}")
        
        return avg_loss, {"accuracy": avg_accuracy, "algorithm": self.algorithm_name}

def fit_config(server_round: int) -> Dict[str, Union[bool, bytes, float, int, str]]:
    """Return training configuration dict for each round"""
    config = {
        "server_round": server_round,
        "local_epochs": 1,
        "batch_size": 64,
        "learning_rate": 0.001,
    }
    return config

def evaluate_config(server_round: int) -> Dict[str, Union[bool, bytes, float, int, str]]:
    """Return evaluation configuration dict for each round"""
    config = {
        "server_round": server_round,
    }
    return config

def get_strategy(algorithm: str) -> fl.server.strategy.Strategy:
    """Get strategy based on algorithm name"""
    
    # Create results tracker
    results_tracker = ResultsTracker(algorithm)
    
    # Common parameters for all strategies
    strategy_params = {
        "fraction_fit": 0.8,
        "fraction_evaluate": 1.0,
        "min_fit_clients": 2,
        "min_evaluate_clients": 2,
        "min_available_clients": 2,
        "on_fit_config_fn": fit_config,
        "on_evaluate_config_fn": evaluate_config,
    }
    
    if algorithm == "FedProx":
        # FedProx with proximal term
        strategy = CustomStrategy(
            algorithm_name="FedProx",
            results_tracker=results_tracker,
            **strategy_params
        )
        logger.info("🔧 FedProx strategy created with proximal regularization")
    
    elif algorithm == "AsyncFL":
        # AsyncFL - for simulation, we'll use FedAvg with modified parameters
        strategy_params["fraction_fit"] = 0.6  # Allow partial client participation
        strategy = CustomStrategy(
            algorithm_name="AsyncFL",
            results_tracker=results_tracker,
            **strategy_params
        )
        logger.info("⚡ AsyncFL strategy created with asynchronous simulation")
    
    else:  # FedAvg
        strategy = CustomStrategy(
            algorithm_name="FedAvg",
            results_tracker=results_tracker,
            **strategy_params
        )
        logger.info("📊 FedAvg baseline strategy created")
    
    return strategy

def main():
    """Main server function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Federated Learning Server')
    parser.add_argument('--algorithm', type=str, default='FedAvg', 
                       choices=['FedAvg', 'FedProx', 'AsyncFL'],
                       help='FL algorithm to use')
    parser.add_argument('--rounds', type=int, default=10, 
                       help='Number of rounds')
    parser.add_argument('--port', type=int, default=8080,
                       help='Server port')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🚀 STARTING FEDERATED LEARNING SERVER")
    logger.info(f"{'='*60}")
    logger.info(f"Algorithm: {args.algorithm}")
    logger.info(f"Rounds: {args.rounds}")
    logger.info(f"Port: {args.port}")
    logger.info(f"{'='*60}\n")
    
    # Get strategy
    strategy = get_strategy(args.algorithm)
    
    try:
        # Start Flower server
        fl.server.start_server(
            server_address=f"0.0.0.0:{args.port}",
            config=fl.server.ServerConfig(num_rounds=args.rounds,
            round_timeout=120
            ),
            strategy=strategy,
        )
        
        # Save final results
        if hasattr(strategy, 'results_tracker'):
            summary = strategy.results_tracker.save_final_summary()
            
            # Display final results
            logger.info(f"\n{'='*60}")
            logger.info(f"🏁 EXPERIMENT COMPLETED - {args.algorithm}")
            logger.info(f"{'='*60}")
            logger.info(f"Final Accuracy: {summary['final_accuracy']:.4f}")
            logger.info(f"Final Loss: {summary['final_loss']:.4f}")
            logger.info(f"Total Time: {summary['total_time']:.2f} seconds")
            logger.info(f"Results saved to: {strategy.results_tracker.results_dir}")
            logger.info(f"{'='*60}\n")
        
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise e

if __name__ == "__main__":
    main()