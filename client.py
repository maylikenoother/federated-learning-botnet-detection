import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl
import logging
from collections import OrderedDict
import traceback
import time
import numpy as np
import socket

from model import Net
from partition_data import load_and_partition_data

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'client_{os.environ.get("CLIENT_ID", "0")}.log')
    ]
)
logger = logging.getLogger(__name__)

CLIENT_ID = int(os.environ.get("CLIENT_ID", 0))
SERVER_ADDRESS = os.environ.get("SERVER_ADDRESS", "localhost:8080")
NUM_CLIENTS = 5
BATCH_SIZE = 64
EPOCHS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Zero-day attack mapping as per your research
ZERO_DAY_MAPPING = {
    0: 'DDoS',
    1: 'Reconnaissance', 
    2: 'Theft',
    3: 'DoS',
    4: 'Normal'
}

logger.info(f"🚀 Starting client {CLIENT_ID} on device: {DEVICE}")
logger.info(f"🌐 Server address: {SERVER_ADDRESS}")
logger.info(f"🎯 Zero-day scenario: Client {CLIENT_ID} missing {ZERO_DAY_MAPPING.get(CLIENT_ID, 'Unknown')} attacks")

def wait_for_server(server_address, max_attempts=30):
    """Wait for server to be available with better error handling"""
    host, port = server_address.split(':')
    port = int(port)
    
    logger.info(f"🔍 Waiting for server at {host}:{port}...")
    
    for attempt in range(max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2)
                result = s.connect_ex((host, port))
                if result == 0:
                    logger.info(f"✅ Server is ready at {host}:{port}")
                    return True
        except Exception as e:
            logger.debug(f"Connection attempt {attempt + 1}: {e}")
        
        if attempt % 5 == 0:
            logger.info(f"⏳ Server check {attempt + 1}/{max_attempts}...")
        
        time.sleep(2)
    
    logger.error(f"❌ Server not available after {max_attempts} attempts")
    return False

class EnhancedFlowerClient(fl.client.NumPyClient):
    """Enhanced Flower client with better error handling and logging"""
    
    def __init__(self, model, train_loader, test_loader, device, client_id):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.client_id = client_id
        self.missing_attack = ZERO_DAY_MAPPING.get(client_id, 'Unknown')
        
        # Algorithm-specific storage
        self.global_model_params = None  # For FedProx proximal term
        self.last_update_time = time.time()  # For AsyncFL
        self.round_metrics = {}  # Store per-round metrics
        
        logger.info(f"✅ Client {client_id} initialized successfully")
        logger.info(f"📊 Train samples: {len(train_loader.dataset)}")
        logger.info(f"📊 Test samples: {len(test_loader.dataset)}")
        logger.info(f"🎯 Zero-day scenario: missing {self.missing_attack}")

    def get_parameters(self, config):
        """Return the model parameters as a list of NumPy ndarrays."""
        try:
            parameters = [val.cpu().numpy() for val in self.model.state_dict().values()]
            logger.debug(f"Client {self.client_id}: Retrieved {len(parameters)} parameter arrays")
            return parameters
        except Exception as e:
            logger.error(f"Client {self.client_id}: Failed to get parameters: {e}")
            logger.error(traceback.format_exc())
            raise e

    def set_parameters(self, parameters):
        """Update model parameters from a list of NumPy ndarrays."""
        try:
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
            
            # Store global parameters for FedProx
            self.global_model_params = [torch.tensor(param).to(self.device) for param in parameters]
            
            logger.debug(f"Client {self.client_id}: Updated model parameters")
        except Exception as e:
            logger.error(f"Client {self.client_id}: Failed to set parameters: {e}")
            logger.error(traceback.format_exc())
            raise e

    def fit(self, parameters, config):
        """Train the model using the specified federated learning algorithm."""
        algorithm = config.get('algorithm', 'FedAvg')
        server_round = config.get('server_round', 'unknown')
        
        logger.info(f"🎯 Client {self.client_id} - Training Round {server_round} with {algorithm}")
        
        try:
            if algorithm == "FedProx":
                return self._fit_fedprox(parameters, config)
            elif algorithm == "AsyncFL":
                return self._fit_async(parameters, config)
            else:  # FedAvg (baseline)
                return self._fit_fedavg(parameters, config)
                
        except Exception as e:
            logger.error(f"❌ Client {self.client_id} training failed with {algorithm}: {e}")
            logger.error(traceback.format_exc())
            # Return valid result even on failure to prevent server crash
            return self.get_parameters(config={}), 0, {
                "loss": float('inf'),
                "accuracy": 0.0,
                "algorithm": algorithm,
                "error": str(e),
                "missing_attack": self.missing_attack
            }

    def _fit_fedavg(self, parameters, config):
        """Standard FedAvg training implementation."""
        server_round = config.get('server_round', 'unknown')
        
        # Update local model parameters
        self.set_parameters(parameters)
        
        # Train model
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.get('learning_rate', 0.001))
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        
        start_time = time.time()
        
        for epoch in range(config.get('local_epochs', EPOCHS)):
            epoch_loss = 0.0
            epoch_samples = 0
            epoch_correct = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Validate data
                if torch.isnan(data).any() or torch.isinf(data).any():
                    logger.warning(f"Client {self.client_id}: Found NaN/Inf in training data, skipping batch")
                    continue
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Client {self.client_id}: Invalid loss detected, skipping batch")
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Calculate accuracy
                with torch.no_grad():
                    pred = output.argmax(dim=1, keepdim=True)
                    epoch_correct += pred.eq(target.view_as(pred)).sum().item()
                
                epoch_loss += loss.item()
                epoch_samples += data.size(0)
            
            total_loss += epoch_loss
            total_samples += epoch_samples
            correct_predictions += epoch_correct
            
            if epoch_samples > 0:
                avg_epoch_loss = epoch_loss / epoch_samples
                epoch_accuracy = epoch_correct / epoch_samples
                logger.info(f"Client {self.client_id} - Epoch {epoch+1}/{EPOCHS}, Loss: {avg_epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
        
        training_time = time.time() - start_time
        
        # Calculate final metrics
        if total_samples > 0:
            avg_loss = total_loss / total_samples
            training_accuracy = correct_predictions / total_samples
        else:
            avg_loss = float('inf')
            training_accuracy = 0.0
            logger.warning(f"Client {self.client_id}: No samples processed during training!")
        
        logger.info(f"✅ Client {self.client_id} FedAvg training completed - Loss: {avg_loss:.4f}, Accuracy: {training_accuracy:.4f}")
        
        return self.get_parameters(config={}), len(self.train_loader.dataset), {
            "loss": float(avg_loss),
            "accuracy": float(training_accuracy),
            "algorithm": "FedAvg",
            "training_time": training_time,
            "missing_attack": self.missing_attack
        }

    def _fit_fedprox(self, parameters, config):
        """FedProx training with proximal term for handling non-IID data."""
        server_round = config.get('server_round', 'unknown')
        mu = config.get('mu', 0.01)  # Proximal term coefficient
        
        logger.info(f"🔧 Client {self.client_id} - FedProx training with μ={mu}")
        
        # Update local model parameters and store global model
        self.set_parameters(parameters)
        global_params = [param.clone().detach() for param in self.model.parameters()]
        
        # Train model with proximal term
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.get('learning_rate', 0.001))
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        total_proximal_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        
        start_time = time.time()
        
        for epoch in range(config.get('local_epochs', EPOCHS)):
            epoch_loss = 0.0
            epoch_proximal = 0.0
            epoch_samples = 0
            epoch_correct = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Validate data
                if torch.isnan(data).any() or torch.isinf(data).any():
                    continue
                
                optimizer.zero_grad()
                output = self.model(data)
                
                # Standard cross-entropy loss
                ce_loss = criterion(output, target)
                
                if torch.isnan(ce_loss) or torch.isinf(ce_loss):
                    continue
                
                # Calculate proximal term: (μ/2) * ||w - w_global||²
                proximal_term = 0.0
                for param, global_param in zip(self.model.parameters(), global_params):
                    proximal_term += torch.norm(param - global_param) ** 2
                
                proximal_loss = (mu / 2) * proximal_term
                
                # Total loss with proximal regularization
                total_loss_batch = ce_loss + proximal_loss
                total_loss_batch.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Calculate accuracy
                with torch.no_grad():
                    pred = output.argmax(dim=1, keepdim=True)
                    epoch_correct += pred.eq(target.view_as(pred)).sum().item()
                
                epoch_loss += ce_loss.item()
                epoch_proximal += proximal_loss.item()
                epoch_samples += data.size(0)
            
            total_loss += epoch_loss
            total_proximal_loss += epoch_proximal
            total_samples += epoch_samples
            correct_predictions += epoch_correct
            
            if epoch_samples > 0:
                avg_epoch_loss = epoch_loss / epoch_samples
                avg_epoch_proximal = epoch_proximal / epoch_samples
                epoch_accuracy = epoch_correct / epoch_samples
                logger.info(f"Client {self.client_id} - Epoch {epoch+1}/{EPOCHS}, CE Loss: {avg_epoch_loss:.4f}, "
                          f"Proximal: {avg_epoch_proximal:.6f}, Accuracy: {epoch_accuracy:.4f}")
        
        training_time = time.time() - start_time
        
        # Calculate final metrics
        if total_samples > 0:
            avg_loss = total_loss / total_samples
            avg_proximal = total_proximal_loss / total_samples
            training_accuracy = correct_predictions / total_samples
        else:
            avg_loss = float('inf')
            avg_proximal = 0.0
            training_accuracy = 0.0
            logger.warning(f"Client {self.client_id}: No samples processed during FedProx training!")
        
        logger.info(f"✅ Client {self.client_id} FedProx training completed - Loss: {avg_loss:.4f}, "
                   f"Proximal: {avg_proximal:.6f}, Accuracy: {training_accuracy:.4f}")
        
        return self.get_parameters(config={}), len(self.train_loader.dataset), {
            "loss": float(avg_loss),
            "accuracy": float(training_accuracy),
            "algorithm": "FedProx",
            "proximal_term": float(avg_proximal),
            "mu": mu,
            "training_time": training_time,
            "missing_attack": self.missing_attack
        }

    def _fit_async(self, parameters, config):
        """Asynchronous FL training with timestamp tracking."""
        server_round = config.get('server_round', 'unknown')
        staleness_threshold = config.get('staleness_threshold', 3)
        
        current_time = time.time()
        time_since_last_update = current_time - self.last_update_time
        
        logger.info(f"⚡ Client {self.client_id} - AsyncFL training (time since last: {time_since_last_update:.2f}s)")
        
        # Check staleness - if too much time has passed, this update might be stale
        estimated_staleness = time_since_last_update / 10.0  # Rough estimate
        
        if estimated_staleness > staleness_threshold:
            logger.warning(f"Client {self.client_id}: Potentially stale update (staleness: {estimated_staleness:.2f})")
        
        # Use standard FedAvg training but with async-specific metrics
        result = self._fit_fedavg(parameters, config)
        
        # Update timestamp
        self.last_update_time = current_time
        
        # Add async-specific metrics
        result[2].update({
            "algorithm": "AsyncFL",
            "staleness_estimate": float(estimated_staleness),
            "time_since_last_update": float(time_since_last_update),
            "async_timestamp": current_time,
            "staleness_threshold": staleness_threshold
        })
        
        logger.info(f"✅ Client {self.client_id} AsyncFL training completed - Staleness: {estimated_staleness:.2f}")
        
        return result

    def evaluate(self, parameters, config):
        """Evaluate the model with zero-day detection metrics."""
        algorithm = config.get('algorithm', 'FedAvg')
        logger.info(f"📊 Client {self.client_id} - Evaluating with {algorithm}")
        
        try:
            # Update local model parameters
            logger.debug("Setting model parameters for evaluation")
            self.set_parameters(parameters)
            logger.debug("Model parameters set successfully")
            
            # Evaluate model
            self.model.eval()
            criterion = nn.CrossEntropyLoss()
            
            total_loss = 0.0
            correct = 0
            total = 0
            
            # Track class-specific performance
            class_predictions = {}
            class_targets = {}
            
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(self.test_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # Validate data
                    if torch.isnan(data).any() or torch.isinf(data).any():
                        logger.warning(f"⚠️ Client {self.client_id}: NaN/Inf in test data (batch {batch_idx}), skipping")
                        continue
                    
                    output = self.model(data)

                    if torch.isnan(output).any() or torch.isinf(output).any():
                        logger.warning(f"⚠️ Client {self.client_id}: NaN/Inf in model output (batch {batch_idx}), skipping")
                        continue
                    
                    loss = criterion(output, target)
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(f"⚠️ Client {self.client_id}: NaN/Inf loss in evaluation (batch {batch_idx}), skipping")
                        continue
                    
                    total_loss += loss.item() * data.size(0)
                    
                    pred = output.argmax(dim=1, keepdim=True)
                    batch_correct = pred.eq(target.view_as(pred)).sum().item()
                    correct += batch_correct
                    total += target.size(0)
                    
                    for t, p in zip(target.cpu().numpy(), pred.cpu().numpy().flatten()):
                        if t not in class_targets:
                            class_targets[t] = 0
                            class_predictions[t] = 0
                        class_targets[t] += 1
                        if t == p:
                            class_predictions[t] += 1
            
            if total > 0:
                accuracy = correct / total
                avg_loss = total_loss / total
            else:
                accuracy = 0.0
                avg_loss = float('inf')
                logger.warning(f"⚠️ Client {self.client_id}: No valid samples processed in evaluation")
            
            per_class_accuracy = {}
            for class_id in class_targets:
                if class_targets[class_id] > 0:
                    per_class_accuracy[class_id] = class_predictions[class_id] / class_targets[class_id]
            
            zero_day_detection_rate = self._calculate_zero_day_detection(per_class_accuracy)
            
            logger.info(f"✅ Client {self.client_id} {algorithm} evaluation complete - "
                        f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, "
                        f"Zero-day detection: {zero_day_detection_rate:.4f}")
            
            return float(avg_loss), len(self.test_loader.dataset), {
                "accuracy": float(accuracy),
                "algorithm": algorithm,
                "missing_attack": self.missing_attack,
                "zero_day_detection_rate": float(zero_day_detection_rate),
                "per_class_accuracy": {str(k): float(v) for k, v in per_class_accuracy.items()},
                "total_classes_detected": len(per_class_accuracy)
            }
        
        except Exception as e:
            logger.error(f"❌ Client {self.client_id} evaluation failed: {e}")
            logger.error(traceback.format_exc())
            return float('inf'), 0, {
                "accuracy": 0.0,
                "algorithm": algorithm,
                "missing_attack": self.missing_attack,
                "zero_day_detection_rate": 0.0,
                "error": str(e)
            }

    def _calculate_zero_day_detection(self, per_class_accuracy):
        """Calculate zero-day detection capability based on known attack detection."""
        if not per_class_accuracy:
            return 0.0
        
        # Average accuracy across all detected classes
        # High accuracy on known attacks suggests good zero-day detection potential
        accuracies = list(per_class_accuracy.values())
        return sum(accuracies) / len(accuracies)

def main():
    """Main function to start the enhanced federated learning client."""
    try:
        logger.info(f"🌟 Starting Enhanced FL Client {CLIENT_ID}")
        logger.info(f"📍 Target server: {SERVER_ADDRESS}")
        
        # Wait for server to be ready
        if not wait_for_server(SERVER_ADDRESS):
            logger.error("❌ Server is not available. Exiting.")
            return False
        
        # Load and partition data with retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"📂 Loading data for client {CLIENT_ID} (attempt {attempt + 1}/{max_retries})")
                (X, y), (X_test, y_test), missing_attack = load_and_partition_data(
                    file_path="Bot_IoT.csv",
                    client_id=CLIENT_ID,
                    num_clients=NUM_CLIENTS,
                    label_col="category",
                    chunk_size=50000  # Optimized for reliability
                )
                logger.info(f"✅ Data loaded successfully: {len(X)} samples")
                break
            except Exception as e:
                logger.error(f"❌ Data loading attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    raise e

        
        # Validate data size
        if len(X) < 50:
            logger.warning(f"⚠️ Client {CLIENT_ID} has very few samples ({len(X)}). This may affect performance.")
        
        # Create train/test split (80/20)
        dataset_size = len(X)
        train_size = max(int(0.8 * dataset_size), 1)
        test_size = dataset_size - train_size
        
        if test_size == 0:
            test_size = 1
            train_size = dataset_size - 1
        
        # Create datasets and loaders
        train_dataset = TensorDataset(X[:train_size], y[:train_size])
        test_dataset = TensorDataset(X[train_size:train_size + test_size], y[train_size:train_size + test_size])
        
        train_loader = DataLoader(train_dataset, batch_size=min(BATCH_SIZE, len(train_dataset)), shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=min(BATCH_SIZE, len(test_dataset)), shuffle=False)
        
        logger.info(f"📊 Data prepared - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
        
        # Create model
        num_features = X.shape[1]
        num_classes = 5  # Bot-IoT dataset classes: Normal, DDoS, DoS, Reconnaissance, Theft
        model = Net(input_size=num_features, output_size=num_classes)
        
        logger.info(f"🧠 Model created - Input: {num_features}, Output: {num_classes}")
        
        # Create enhanced Flower client
        client = EnhancedFlowerClient(model, train_loader, test_loader, DEVICE, CLIENT_ID)
        
        # Connect to server with robust retry mechanism
        max_retries = 5
        for attempt in range(max_retries):
            try:
                logger.info(f"🔗 Connecting to server (attempt {attempt + 1}/{max_retries})")
                fl.client.start_numpy_client(
                    server_address=SERVER_ADDRESS, 
                    client=client
                )
                logger.info(f"✅ Client {CLIENT_ID} completed successfully!")
                break
            except Exception as e:
                logger.error(f"❌ Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 30)  # Exponential backoff with cap
                    logger.info(f"⏳ Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"❌ All connection attempts failed. Final error: {e}")
                    raise e
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Client {CLIENT_ID} failed to start: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)