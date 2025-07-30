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
import threading
import time
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedResultsTracker:
    def __init__(self, algorithm_name="FedAvg", experiment_name=None):
        self.algorithm_name = algorithm_name
        self.experiment_name = experiment_name or f"{algorithm_name}_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_dir = f"results/{self.experiment_name}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Enhanced metrics storage for comparison
        self.training_history = []
        self.evaluation_history = []
        self.communication_rounds = []
        self.convergence_metrics = []
        self.client_metrics = {}
        
        # Communication efficiency tracking
        self.bytes_transmitted = 0
        self.total_communication_time = 0
        self.gradient_divergence = []
        
        # Initialize CSV files
        self._init_csv_files()
        
        logger.info(f"📊 Enhanced results tracker initialized for {algorithm_name}")
        logger.info(f"📂 Results will be saved to: {self.results_dir}")
    
    def _init_csv_files(self):
        """Initialize enhanced CSV files for comprehensive analysis"""
        # Communication efficiency tracking
        self.comm_csv = os.path.join(self.results_dir, "communication_efficiency.csv")
        with open(self.comm_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['round', 'bytes_transmitted', 'communication_time', 'num_participants', 
                           'convergence_rate', 'gradient_norm', 'timestamp'])
        
        # Convergence analysis
        self.convergence_csv = os.path.join(self.results_dir, "convergence_analysis.csv")
        with open(self.convergence_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['round', 'loss_improvement', 'accuracy_improvement', 'convergence_rate', 
                           'time_to_target', 'communication_rounds_to_target', 'timestamp'])
    
    def log_communication_round(self, round_num, bytes_sent, comm_time, participants, convergence_rate=None, gradient_norm=None):
        """Log communication efficiency metrics"""
        timestamp = datetime.now().isoformat()
        
        self.bytes_transmitted += bytes_sent
        self.total_communication_time += comm_time
        
        # Save to CSV
        with open(self.comm_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([round_num, bytes_sent, comm_time, participants, 
                           convergence_rate or 0.0, gradient_norm or 0.0, timestamp])
        
        # Display communication efficiency
        print(f"\n📡 COMMUNICATION ROUND {round_num}")
        print(f"Bytes Transmitted: {bytes_sent:,}")
        print(f"Communication Time: {comm_time:.3f}s")
        print(f"Cumulative Bytes: {self.bytes_transmitted:,}")
        print(f"Total Comm Time: {self.total_communication_time:.3f}s")
        if convergence_rate:
            print(f"Convergence Rate: {convergence_rate:.6f}")

# FedProx Strategy Implementation
class FedProxStrategy(fl.server.strategy.FedAvg):
    """
    FedProx implementation with proximal term to handle non-IID data
    Based on Li et al. (2020) - Federated Optimization in Heterogeneous Networks
    """
    def __init__(self, mu=0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = mu  # Proximal term coefficient
        self.current_round = 0
        self.global_model_params = None
        
        logger.info(f"🔧 FedProx Strategy initialized with μ={mu}")
    
    def aggregate_fit(self, server_round, results, failures):
        """Enhanced aggregation with proximal term tracking"""
        self.current_round = server_round
        
        # Log communication metrics
        if hasattr(self, 'results_tracker'):
            bytes_sent = sum(len(str(r.parameters)) * 4 for _, r in results)  # Approximate
            comm_time = time.time() - getattr(self, '_round_start_time', time.time())
            self.results_tracker.log_communication_round(
                server_round, bytes_sent, comm_time, len(results)
            )
        
        # Standard FedAvg aggregation with proximal term consideration
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        # Store global model for next round's proximal term
        self.global_model_params = aggregated_parameters
        
        # Enhanced metrics for FedProx analysis
        if aggregated_metrics:
            aggregated_metrics['algorithm'] = 'FedProx'
            aggregated_metrics['mu'] = self.mu
            aggregated_metrics['proximal_term_active'] = True
        
        return aggregated_parameters, aggregated_metrics

# Asynchronous FL Strategy Implementation  
class AsyncFLStrategy(fl.server.strategy.Strategy):
    """
    Asynchronous Federated Learning implementation
    Based on Xie et al. (2019) - Asynchronous Federated Optimization
    """
    def __init__(self, min_clients=2, staleness_threshold=3):
        self.min_clients = min_clients
        self.staleness_threshold = staleness_threshold
        self.global_model = None
        self.client_updates = {}
        self.round_timestamps = defaultdict(float)
        self.lock = threading.Lock()
        
        logger.info(f"⚡ AsyncFL Strategy initialized with staleness_threshold={staleness_threshold}")
    
    def initialize_parameters(self, client_manager):
        """Initialize global model parameters"""
        return fl.common.ndarrays_to_parameters([])
    
    def configure_fit(self, server_round, parameters, client_manager):
        """Configure clients for asynchronous training"""
        # In async FL, clients can train whenever they're ready
        sample_size = max(self.min_clients, int(len(client_manager.all()) * 0.6))
        clients = client_manager.sample(num_clients=sample_size)
        
        config = {
            "server_round": server_round,
            "local_epochs": 1,
            "learning_rate": 0.001,
            "async_mode": True,
            "staleness_threshold": self.staleness_threshold
        }
        
        return [(client, fl.common.FitIns(parameters, config)) for client in clients]
    
    def aggregate_fit(self, server_round, results, failures):
        """Asynchronous aggregation with staleness handling"""
        current_time = time.time()
        
        with self.lock:
            # Filter out stale updates
            fresh_results = []
            for client_id, result in enumerate(results):
                client_key = f"client_{client_id}"
                staleness = server_round - self.round_timestamps.get(client_key, server_round)
                
                if staleness <= self.staleness_threshold:
                    fresh_results.append((client_id, result))
                    self.round_timestamps[client_key] = server_round
                else:
                    logger.warning(f"Discarding stale update from {client_key} (staleness: {staleness})")
            
            if not fresh_results:
                logger.warning("No fresh updates available for aggregation")
                return self.global_model, {}
            
            # Weighted aggregation based on data size and staleness
            aggregated_ndarrays = self._aggregate_async(fresh_results)
            
            # Update global model
            self.global_model = fl.common.ndarrays_to_parameters(aggregated_ndarrays)
            
            # Enhanced metrics
            metrics = {
                "algorithm": "AsyncFL",
                "fresh_updates": len(fresh_results),
                "discarded_stale": len(results) - len(fresh_results),
                "avg_staleness": np.mean([server_round - self.round_timestamps.get(f"client_{i}", server_round) 
                                        for i, _ in fresh_results])
            }
            
            return self.global_model, metrics
    
    def _aggregate_async(self, results):
        """Perform weighted aggregation for async updates"""
        if not results:
            return []
        
        # Extract parameters and weights
        weights_results = [(result.num_examples, fl.common.parameters_to_ndarrays(result.parameters)) 
                          for _, result in results]
        
        if not weights_results:
            return []
        
        # Weighted average
        total_examples = sum(num_examples for num_examples, _ in weights_results)
        
        # Initialize aggregated parameters
        aggregated_ndarrays = [
            np.zeros_like(layer) for layer in weights_results[0][1]
        ]
        
        # Weighted sum
        for num_examples, ndarrays in weights_results:
            weight = num_examples / total_examples
            for i, layer in enumerate(ndarrays):
                aggregated_ndarrays[i] += weight * layer
        
        return aggregated_ndarrays

# Enhanced client configuration for different algorithms
def enhanced_fit_config(server_round: int, algorithm: str = "FedAvg") -> Dict[str, Union[bool, bytes, float, int, str]]:
    """Return algorithm-specific training configuration"""
    base_config = {
        "server_round": server_round,
        "local_epochs": 1,
        "batch_size": 64,
        "learning_rate": 0.001,
        "algorithm": algorithm,
    }
    
    if algorithm == "FedProx":
        base_config.update({
            "mu": 0.01,  # Proximal term coefficient
            "proximal_term": True,
        })
    elif algorithm == "AsyncFL":
        base_config.update({
            "async_mode": True,
            "staleness_threshold": 3,
        })
    
    return base_config

# Main experiment runner
def run_federated_experiment(algorithm="FedAvg", num_rounds=10):
    """Run federated learning experiment with specified algorithm"""
    logger.info(f"🚀 Starting Federated Learning Experiment: {algorithm}")
    
    # Initialize results tracker
    results_tracker = EnhancedResultsTracker(algorithm_name=algorithm)
    
    # Select strategy based on algorithm
    if algorithm == "FedProx":
        strategy = FedProxStrategy(
            mu=0.01,
            fraction_fit=0.8,
            fraction_evaluate=1.0,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
        )
    elif algorithm == "AsyncFL":
        strategy = AsyncFLStrategy(
            min_clients=2,
            staleness_threshold=3
        )
    else:  # FedAvg (baseline)
        from server import CustomStrategy  # Use your existing strategy
        strategy = CustomStrategy(
            fraction_fit=0.8,
            fraction_evaluate=1.0,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
        )
    
    # Attach results tracker to strategy
    strategy.results_tracker = results_tracker
    
    try:
        # Start server with algorithm-specific configuration
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
        )
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
    finally:
        # Save results and generate comparison metrics
        results_tracker.save_final_summary()
        generate_algorithm_comparison(algorithm, results_tracker)
        logger.info(f"📊 {algorithm} experiment completed!")

def generate_algorithm_comparison(algorithm, results_tracker):
    """Generate comparative analysis metrics"""
    comparison_file = f"results/comparison_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Calculate key performance indicators
    metrics = {
        "algorithm": algorithm,
        "total_communication_rounds": len(results_tracker.evaluation_history),
        "total_bytes_transmitted": results_tracker.bytes_transmitted,
        "total_communication_time": results_tracker.total_communication_time,
        "final_accuracy": results_tracker.evaluation_history[-1]['accuracy'] if results_tracker.evaluation_history else 0.0,
        "convergence_rate": calculate_convergence_rate(results_tracker.evaluation_history),
        "communication_efficiency": results_tracker.bytes_transmitted / max(results_tracker.total_communication_time, 1),
        "rounds_to_95_percent": find_rounds_to_target_accuracy(results_tracker.evaluation_history, 0.95),
        "timestamp": datetime.now().isoformat()
    }
    
    # Save comparison metrics
    with open(comparison_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"📈 Comparison metrics saved to: {comparison_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"🎯 {algorithm.upper()} PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"Final Accuracy: {metrics['final_accuracy']:.4f}")
    print(f"Total Rounds: {metrics['total_communication_rounds']}")
    print(f"Bytes Transmitted: {metrics['total_bytes_transmitted']:,}")
    print(f"Communication Time: {metrics['total_communication_time']:.2f}s")
    print(f"Rounds to 95% Accuracy: {metrics['rounds_to_95_percent']}")
    print(f"Communication Efficiency: {metrics['communication_efficiency']:.2f} bytes/s")
    print(f"{'='*60}\n")

def calculate_convergence_rate(evaluation_history):
    """Calculate convergence rate based on accuracy improvements"""
    if len(evaluation_history) < 2:
        return 0.0
    
    accuracies = [round_data['accuracy'] for round_data in evaluation_history]
    improvements = [accuracies[i] - accuracies[i-1] for i in range(1, len(accuracies))]
    
    # Average improvement per round
    return np.mean(improvements) if improvements else 0.0

def find_rounds_to_target_accuracy(evaluation_history, target_accuracy=0.95):
    """Find number of rounds needed to reach target accuracy"""
    for i, round_data in enumerate(evaluation_history):
        if round_data['accuracy'] >= target_accuracy:
            return i + 1
    return len(evaluation_history)  # If target not reached

if __name__ == "__main__":
    # Run experiments for comparison
    algorithms = ["FedAvg", "FedProx", "AsyncFL"]
    
    for algorithm in algorithms:
        print(f"\n🔄 Starting {algorithm} experiment...")
        run_federated_experiment(algorithm=algorithm, num_rounds=10)
        print(f"✅ {algorithm} experiment completed!")
        
        # Brief pause between experiments
        time.sleep(5)
    
    print("\n🎉 All federated learning algorithms tested!")
    print("📊 Check the results/ directory for detailed comparisons.")


# Enhanced client.py modifications for algorithm support
"""
Add this to your client.py to support different algorithms:

def fit(self, parameters, config):
    algorithm = config.get('algorithm', 'FedAvg')
    server_round = config.get('server_round', 'unknown')
    
    logger.info(f"🎯 Client {self.client_id} - Training Round {server_round} with {algorithm}")
    
    if algorithm == "FedProx":
        return self._fit_fedprox(parameters, config)
    elif algorithm == "AsyncFL":
        return self._fit_async(parameters, config)
    else:
        return self._fit_fedavg(parameters, config)  # Your existing implementation

def _fit_fedprox(self, parameters, config):
    # FedProx-specific training with proximal term
    self.set_parameters(parameters)
    
    # Store global model for proximal term
    global_params = [param.clone() for param in self.model.parameters()]
    
    self.model.train()
    optimizer = torch.optim.Adam(self.model.parameters(), lr=config.get('learning_rate', 0.001))
    criterion = nn.CrossEntropyLoss()
    mu = config.get('mu', 0.01)
    
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    for epoch in range(config.get('local_epochs', 1)):
        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            
            # Standard loss
            loss = criterion(output, target)
            
            # Add proximal term
            proximal_term = 0.0
            for param, global_param in zip(self.model.parameters(), global_params):
                proximal_term += torch.norm(param - global_param) ** 2
            
            total_loss_with_prox = loss + (mu / 2) * proximal_term
            total_loss_with_prox.backward()
            
            optimizer.step()
            
            # Track metrics
            with torch.no_grad():
                pred = output.argmax(dim=1)
                correct_predictions += pred.eq(target).sum().item()
            
            total_loss += loss.item()
            total_samples += data.size(0)
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    
    return self.get_parameters({}), len(self.train_loader.dataset), {
        "loss": float(avg_loss),
        "accuracy": float(accuracy),
        "algorithm": "FedProx",
        "proximal_term": float(proximal_term),
        "missing_attack": self.missing_attack
    }

def _fit_async(self, parameters, config):
    # Async FL with timestamp tracking
    start_time = time.time()
    result = self._fit_fedavg(parameters, config)  # Use standard training
    training_time = time.time() - start_time
    
    # Add async-specific metrics
    result[2].update({
        "algorithm": "AsyncFL",
        "training_time": training_time,
        "async_timestamp": time.time()
    })
    
    return result
"""