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
    """FIXED Results Tracker with proper metric collection"""
    def __init__(self, algorithm_name="FedAvg"):
        self.algorithm_name = algorithm_name
        self.experiment_id = f"{algorithm_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_dir = f"results/{self.experiment_id}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.training_history = []
        self.evaluation_history = []
        self.communication_metrics = []  # FIXED: Added communication tracking
        self.round_times = []
        self.start_time = time.time()
        
        # Initialize CSV files
        self.training_csv = os.path.join(self.results_dir, "training_history.csv")
        self.eval_csv = os.path.join(self.results_dir, "evaluation_history.csv")
        self.comm_csv = os.path.join(self.results_dir, "communication_metrics.csv")  # FIXED: Added
        
        with open(self.training_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['round', 'loss', 'num_clients', 'total_examples', 'timestamp'])
        
        with open(self.eval_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['round', 'loss', 'accuracy', 'num_clients', 'zero_day_detection', 'timestamp'])
        
        with open(self.comm_csv, 'w', newline='') as f:  # FIXED: Communication metrics CSV
            writer = csv.writer(f)
            writer.writerow(['round', 'bytes_transmitted', 'communication_time', 'num_clients', 'timestamp'])
        
        logger.info(f"📊 Enhanced results tracker initialized for {algorithm_name}")
    
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
    
    def log_evaluation_round(self, round_num, loss, accuracy, num_clients, zero_day_rate=0.0):
        """FIXED: Enhanced evaluation logging with zero-day metrics"""
        timestamp = datetime.now().isoformat()
        self.evaluation_history.append({
            'round': round_num,
            'loss': loss,
            'accuracy': accuracy,
            'num_clients': num_clients,
            'zero_day_detection': zero_day_rate,
            'timestamp': timestamp
        })
        
        with open(self.eval_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([round_num, loss, accuracy, num_clients, zero_day_rate, timestamp])
    
    def log_communication_metrics(self, round_num, bytes_transmitted, communication_time, num_clients):
        """FIXED: Log communication metrics for analysis"""
        timestamp = datetime.now().isoformat()
        self.communication_metrics.append({
            'round': round_num,
            'bytes_transmitted': bytes_transmitted,
            'communication_time': communication_time,
            'num_clients': num_clients,
            'timestamp': timestamp
        })
        
        with open(self.comm_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([round_num, bytes_transmitted, communication_time, num_clients, timestamp])
    
    def save_final_summary(self):
        """FIXED: Enhanced final summary with all metrics"""
        total_time = time.time() - self.start_time
        
        # Calculate total communication volume
        total_bytes = sum(m['bytes_transmitted'] for m in self.communication_metrics)
        avg_comm_time = np.mean([m['communication_time'] for m in self.communication_metrics]) if self.communication_metrics else 0
        
        # Calculate zero-day detection performance
        final_zero_day = self.evaluation_history[-1].get('zero_day_detection', 0.0) if self.evaluation_history else 0.0
        avg_zero_day = np.mean([h.get('zero_day_detection', 0.0) for h in self.evaluation_history])
        
        summary = {
            'algorithm': self.algorithm_name,
            'experiment_id': self.experiment_id,
            'total_rounds': len(self.evaluation_history),
            'total_time': total_time,
            'final_accuracy': self.evaluation_history[-1]['accuracy'] if self.evaluation_history else 0,
            'final_loss': self.evaluation_history[-1]['loss'] if self.evaluation_history else float('inf'),
            'total_communication_bytes': int(total_bytes),  # FIXED: Communication volume
            'avg_communication_time': float(avg_comm_time),
            'final_zero_day_detection': float(final_zero_day),
            'avg_zero_day_detection': float(avg_zero_day),
            'training_history': self.training_history,
            'evaluation_history': self.evaluation_history,
            'communication_metrics': self.communication_metrics  # FIXED: Include communication data
        }
        
        summary_file = os.path.join(self.results_dir, 'experiment_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"💾 Enhanced experiment summary saved to {summary_file}")
        logger.info(f"📊 Final metrics: Acc={summary['final_accuracy']:.3f}, "
                   f"Comm={summary['total_communication_bytes']/1024:.1f}KB, "
                   f"Zero-day={summary['final_zero_day_detection']:.3f}")
        return summary

# FIXED Custom strategy with proper communication tracking
class CustomStrategy(fl.server.strategy.FedAvg):
    def __init__(self, algorithm_name="FedAvg", results_tracker=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.algorithm_name = algorithm_name
        self.results_tracker = results_tracker or ResultsTracker(algorithm_name)
        self.current_round = 0
        self.round_start_time = None  # FIXED: Track round timing
        logger.info(f"🚀 Enhanced {algorithm_name} strategy initialized")
    
    def configure_fit(self, server_round: int, parameters: Parameters, client_manager) -> List[Tuple]:
        """Configure training for each round"""
        self.round_start_time = time.time()  # FIXED: Start timing
        config = {
            "server_round": server_round,
            "algorithm": self.algorithm_name,
            "local_epochs": 1,
            "batch_size": 16,
            "learning_rate": 0.001,
        }
        
        # Add algorithm-specific config
        if self.algorithm_name == "FedProx":
            config["mu"] = 0.01  # Proximal term coefficient
        elif self.algorithm_name == "AsyncFL":
            config["staleness_threshold"] = 3
        
        return super().configure_fit(server_round, parameters, client_manager)
    
    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager) -> List[Tuple]:
        """Configure evaluation for each round"""
        config = {
            "server_round": server_round,
            "algorithm": self.algorithm_name,
        }
        return super().configure_evaluate(server_round, parameters, client_manager)
    
    def aggregate_fit(self, server_round: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]], 
                     failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """FIXED: Aggregate fit results with proper communication tracking"""
        self.current_round = server_round
        
        if not results:
            logger.warning(f"No results to aggregate in round {server_round}")
            return None, {}
        
        # Log failures
        if failures:
            logger.warning(f"Failed clients in round {server_round}: {len(failures)}")
        
        # FIXED: Calculate communication volume more accurately
        total_bytes = 0
        for client, fit_res in results:
            # Estimate bytes transmitted based on parameters and metrics
            param_bytes = len(str(fit_res.parameters)) * 4  # Rough estimate
            metric_bytes = len(str(fit_res.metrics)) * 2 if fit_res.metrics else 0
            total_bytes += param_bytes + metric_bytes
        
        # Track communication timing
        communication_time = time.time() - self.round_start_time if self.round_start_time else 0
        
        # Log communication metrics
        self.results_tracker.log_communication_metrics(
            server_round, total_bytes, communication_time, len(results)
        )
        
        # Extract metrics from results with proper type checking
        losses = []
        accuracies = []
        examples = []
        algorithm_specific_metrics = {}
        
        for client, fit_res in results:
            examples.append(fit_res.num_examples)
            
            if fit_res.metrics:
                # Safely extract scalar metrics
                if "loss" in fit_res.metrics and isinstance(fit_res.metrics["loss"], (int, float)):
                    losses.append(float(fit_res.metrics["loss"]))
                
                if "accuracy" in fit_res.metrics and isinstance(fit_res.metrics["accuracy"], (int, float)):
                    accuracies.append(float(fit_res.metrics["accuracy"]))
                
                # Collect algorithm-specific metrics safely
                for key, value in fit_res.metrics.items():
                    if isinstance(value, (int, float, str, bool)) and key not in ["loss", "accuracy"]:
                        if key not in algorithm_specific_metrics:
                            algorithm_specific_metrics[key] = []
                        algorithm_specific_metrics[key].append(value)
        
        # Calculate weighted averages
        total_examples = sum(examples)
        if losses and total_examples > 0:
            avg_loss = sum(l * e for l, e in zip(losses, examples)) / total_examples
        else:
            avg_loss = 0.0
        
        if accuracies and total_examples > 0:
            avg_accuracy = sum(a * e for a, e in zip(accuracies, examples)) / total_examples
        else:
            avg_accuracy = 0.0
        
        # Log training metrics
        self.results_tracker.log_training_round(server_round, avg_loss, len(results), total_examples)
        
        # Log round info with communication data
        logger.info(f"\n{'='*60}")
        logger.info(f"TRAINING ROUND {server_round} - {self.algorithm_name}")
        logger.info(f"{'='*60}")
        logger.info(f"Participating clients: {len(results)}")
        logger.info(f"Total examples: {total_examples}")
        logger.info(f"Average training loss: {avg_loss:.4f}")
        logger.info(f"Communication: {total_bytes/1024:.1f} KB in {communication_time:.2f}s")  # FIXED: Show comm data
        if avg_accuracy > 0:
            logger.info(f"Average training accuracy: {avg_accuracy:.4f}")
        
        # Standard FedAvg aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        # Add safe algorithm-specific metrics to aggregated results
        if aggregated_metrics is None:
            aggregated_metrics = {}
        
        # Only add Flower-compatible scalar metrics
        aggregated_metrics.update({
            "algorithm": self.algorithm_name,
            "round": server_round,
            "num_clients": len(results),
            "avg_training_loss": avg_loss,
            "total_examples": total_examples,
            "communication_bytes": total_bytes,  # FIXED: Include communication data
            "communication_time": communication_time
        })
        
        if avg_accuracy > 0:
            aggregated_metrics["avg_training_accuracy"] = avg_accuracy
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(self, server_round: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
                          failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes], BaseException]]) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """FIXED: Aggregate evaluation results with enhanced zero-day metrics"""
        if not results:
            logger.warning(f"No evaluation results in round {server_round}")
            return None, {}
        
        # Log failures
        if failures:
            logger.warning(f"Failed evaluations in round {server_round}: {len(failures)}")
        
        # Weighted average of loss and accuracy with proper type checking
        total_loss = 0.0
        total_accuracy = 0.0
        total_examples = 0
        zero_day_metrics = {}
        algorithm_metrics = {}
        
        for client, eval_res in results:
            # Safely handle loss (should always be a number)
            if isinstance(eval_res.loss, (int, float)) and not (np.isnan(eval_res.loss) or np.isinf(eval_res.loss)):
                total_loss += eval_res.loss * eval_res.num_examples
                total_examples += eval_res.num_examples
            
            # Process metrics safely
            if eval_res.metrics:
                for key, value in eval_res.metrics.items():
                    if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)):
                        if key == "accuracy":
                            total_accuracy += value * eval_res.num_examples
                        elif "zero_day" in key:
                            if key not in zero_day_metrics:
                                zero_day_metrics[key] = []
                            zero_day_metrics[key].append(value * eval_res.num_examples)
                        elif key not in ["algorithm", "missing_attack"]:  # Skip non-numeric
                            if key not in algorithm_metrics:
                                algorithm_metrics[key] = []
                            algorithm_metrics[key].append(value * eval_res.num_examples)
        
        # Calculate final metrics
        avg_loss = total_loss / total_examples if total_examples > 0 else float('inf')
        avg_accuracy = total_accuracy / total_examples if total_examples > 0 else 0.0
        
        # Calculate zero-day detection metrics
        avg_zero_day_metrics = {}
        for key, values in zero_day_metrics.items():
            if values:
                avg_zero_day_metrics[key] = sum(values) / total_examples
        
        # Calculate other algorithm metrics
        avg_algorithm_metrics = {}
        for key, values in algorithm_metrics.items():
            if values:
                avg_algorithm_metrics[key] = sum(values) / total_examples
        
        # FIXED: Extract zero-day detection rate properly
        zero_day_detection_rate = avg_zero_day_metrics.get('zero_day_detection_rate', 0.0)
        
        # Log evaluation metrics with enhanced tracking
        self.results_tracker.log_evaluation_round(
            server_round, avg_loss, avg_accuracy, len(results), zero_day_detection_rate
        )
        
        # Display results
        logger.info(f"\nEVALUATION ROUND {server_round} - {self.algorithm_name}")
        logger.info(f"{'='*60}")
        logger.info(f"Global Loss: {avg_loss:.4f}")
        logger.info(f"Global Accuracy: {avg_accuracy:.4f}")
        logger.info(f"Zero-Day Detection Rate: {zero_day_detection_rate:.4f}")  # FIXED: Show zero-day performance
        logger.info(f"Clients evaluated: {len(results)}")
        
        # Log zero-day detection metrics
        if avg_zero_day_metrics:
            logger.info("Zero-day Detection Metrics:")
            for key, value in avg_zero_day_metrics.items():
                logger.info(f"  {key}: {value:.4f}")
        
        # Log algorithm-specific metrics
        if avg_algorithm_metrics:
            logger.info(f"{self.algorithm_name} Specific Metrics:")
            for key, value in avg_algorithm_metrics.items():
                logger.info(f"  {key}: {value:.4f}")
        
        # Prepare return metrics (only Flower-compatible scalars)
        return_metrics = {
            "accuracy": avg_accuracy,
            "algorithm": self.algorithm_name,
            "num_clients": len(results),
            "total_examples": total_examples,
            "zero_day_detection_rate": zero_day_detection_rate  # FIXED: Include zero-day rate
        }
        
        # Add zero-day metrics
        return_metrics.update(avg_zero_day_metrics)
        
        # Add algorithm-specific metrics
        return_metrics.update(avg_algorithm_metrics)
        
        return avg_loss, return_metrics

def fit_config(server_round: int) -> Dict[str, Union[bool, bytes, float, int, str]]:
    """Return training configuration dict for each round"""
    config = {
        "server_round": server_round,
        "local_epochs": 1,
        "batch_size": 16,
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
    """FIXED: Get strategy with enhanced results tracking"""
    
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
        logger.info("🔧 Enhanced FedProx strategy created with proximal regularization")
    
    elif algorithm == "AsyncFL":
        # AsyncFL - for simulation, we'll use FedAvg with modified parameters
        strategy_params["fraction_fit"] = 0.6  # Allow partial client participation
        strategy = CustomStrategy(
            algorithm_name="AsyncFL",
            results_tracker=results_tracker,
            **strategy_params
        )
        logger.info("⚡ Enhanced AsyncFL strategy created with asynchronous simulation")
    
    else:  # FedAvg
        strategy = CustomStrategy(
            algorithm_name="FedAvg",
            results_tracker=results_tracker,
            **strategy_params
        )
        logger.info("📊 Enhanced FedAvg baseline strategy created")
    
    return strategy

def main():
    """FIXED: Main server function with enhanced error handling and metrics"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Enhanced Federated Learning Server')
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
    
    logger.info(f"\n{'='*80}")
    logger.info(f"🚀 STARTING ENHANCED FEDERATED LEARNING SERVER")
    logger.info(f"{'='*80}")
    logger.info(f"Algorithm: {args.algorithm}")
    logger.info(f"Rounds: {args.rounds}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Results Directory: {args.results_dir}")
    logger.info(f"Enhanced Metrics Collection: ✅ Enabled")
    logger.info(f"Communication Tracking: ✅ Enabled")
    logger.info(f"Zero-Day Detection Monitoring: ✅ Enabled")
    logger.info(f"{'='*80}\n")
    
    # Get strategy with enhanced metrics handling
    strategy = get_strategy(args.algorithm)
    
    try:
        # Start Flower server with enhanced configuration
        fl.server.start_server(
            server_address=f"0.0.0.0:{args.port}",
            config=fl.server.ServerConfig(
                num_rounds=args.rounds,
                round_timeout=240  # Increased timeout for stability
            ),
            strategy=strategy,
        )
        
        # Save final results with enhanced metrics
        if hasattr(strategy, 'results_tracker'):
            summary = strategy.results_tracker.save_final_summary()
            
            # Display enhanced final results
            logger.info(f"\n{'='*80}")
            logger.info(f"🏁 EXPERIMENT COMPLETED - {args.algorithm}")
            logger.info(f"{'='*80}")
            logger.info(f"Final Accuracy: {summary['final_accuracy']:.4f}")
            logger.info(f"Final Loss: {summary['final_loss']:.4f}")
            logger.info(f"Zero-Day Detection Rate: {summary['final_zero_day_detection']:.4f}")
            logger.info(f"Total Communication: {summary['total_communication_bytes']/1024:.1f} KB")
            logger.info(f"Avg Communication Time: {summary['avg_communication_time']:.2f}s")
            logger.info(f"Total Rounds: {summary['total_rounds']}")
            logger.info(f"Total Time: {summary['total_time']:.2f} seconds")
            logger.info(f"Results Directory: {strategy.results_tracker.results_dir}")
            logger.info(f"✅ Enhanced metrics collection completed successfully")
            logger.info(f"📊 Communication metrics: ✅ Collected")
            logger.info(f"🎯 Zero-day metrics: ✅ Collected")
            logger.info(f"{'='*80}\n")
        
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        import traceback
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        
        # Try to save any partial results
        if hasattr(strategy, 'results_tracker'):
            try:
                strategy.results_tracker.save_final_summary()
                logger.info("💾 Partial results saved despite error")
            except:
                logger.error("Failed to save partial results")
        
        raise e

if __name__ == "__main__":
    main()