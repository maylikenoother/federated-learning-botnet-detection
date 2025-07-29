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

from model import Net
from partition_data import load_and_partition_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CLIENT_ID = int(os.environ.get("CLIENT_ID", 0))
NUM_CLIENTS = 5
BATCH_SIZE = 64
EPOCHS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global classes for Bot-IoT dataset
GLOBAL_CLASSES = ['DDoS', 'DoS', 'Normal', 'Reconnaissance', 'Theft']
NUM_GLOBAL_CLASSES = len(GLOBAL_CLASSES)

logger.info(f"🚀 Starting zero-day simulation client {CLIENT_ID} on device: {DEVICE}")

class ZeroDayFlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device, client_id, missing_attack):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.client_id = client_id
        self.missing_attack = missing_attack
        self.missing_attack_idx = GLOBAL_CLASSES.index(missing_attack)
        
        logger.info(f"Client {client_id} initialized - Missing attack: {missing_attack} (index {self.missing_attack_idx})")
        logger.info(f"Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")

    def get_parameters(self, config):
        """Return the model parameters as a list of NumPy ndarrays."""
        try:
            parameters = [val.cpu().numpy() for val in self.model.state_dict().values()]
            return parameters
        except Exception as e:
            logger.error(f"Client {self.client_id}: Failed to get parameters: {e}")
            raise e

    def set_parameters(self, parameters):
        """Update model parameters from a list of NumPy ndarrays."""
        try:
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            logger.error(f"Client {self.client_id}: Failed to set parameters: {e}")
            raise e

    def fit(self, parameters, config):
        """Train the model with local data."""
        server_round = config.get('server_round', 'unknown')
        logger.info(f"🎯 Client {self.client_id} - Training Round {server_round}")
        
        try:
            # Update local model parameters
            self.set_parameters(parameters)
            
            # Train model
            self.model.train()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=config.get('learning_rate', 0.001))
            criterion = nn.CrossEntropyLoss()
            
            total_loss = 0.0
            total_samples = 0
            
            for epoch in range(config.get('local_epochs', EPOCHS)):
                epoch_loss = 0.0
                epoch_samples = 0
                
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = criterion(output, target)
                    
                    # Check for valid loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(f"Client {self.client_id}: Invalid loss detected, skipping batch")
                        continue
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    epoch_samples += data.size(0)
                
                total_loss += epoch_loss
                total_samples += epoch_samples
                
                if epoch_samples > 0:
                    avg_epoch_loss = epoch_loss / epoch_samples
                    logger.info(f"Client {self.client_id} - Epoch {epoch+1}/{EPOCHS}, Loss: {avg_epoch_loss:.4f}")
            
            avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
            
            logger.info(f"Client {self.client_id} training completed - Average loss: {avg_loss:.4f}")
            
            return self.get_parameters(config={}), len(self.train_loader.dataset), {
                "loss": float(avg_loss),
                "missing_attack": self.missing_attack
            }
            
        except Exception as e:
            logger.error(f"Client {self.client_id} training failed: {e}")
            logger.error(traceback.format_exc())
            raise e

    def evaluate(self, parameters, config):
        """Evaluate the model on local data with zero-day attack detection metrics."""
        logger.info(f"📊 Client {self.client_id} - Evaluating (Zero-day: {self.missing_attack})")
        
        try:
            # Update local model parameters
            self.set_parameters(parameters)
            
            # Evaluate model
            self.model.eval()
            criterion = nn.CrossEntropyLoss()
            
            total_loss = 0.0
            correct = 0
            total = 0
            
            # Zero-day specific metrics
            zero_day_tp = 0  # True positives for missing attack
            zero_day_fp = 0  # False positives for missing attack
            zero_day_fn = 0  # False negatives for missing attack
            zero_day_tn = 0  # True negatives for missing attack
            zero_day_samples = 0  # Total samples of missing attack type
            
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(self.test_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    output = self.model(data)
                    loss = criterion(output, target)
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(f"Client {self.client_id}: Invalid loss in evaluation, skipping batch")
                        continue
                    
                    total_loss += loss.item() * data.size(0)
                    
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
                    
                    # Zero-day attack detection metrics
                    for i in range(len(target)):
                        true_label = target[i].item()
                        pred_label = pred[i].item()
                        
                        if true_label == self.missing_attack_idx:
                            # This is a zero-day attack sample
                            zero_day_samples += 1
                            if pred_label == self.missing_attack_idx:
                                zero_day_tp += 1
                            else:
                                zero_day_fn += 1
                        else:
                            # This is not a zero-day attack sample
                            if pred_label == self.missing_attack_idx:
                                zero_day_fp += 1
                            else:
                                zero_day_tn += 1
            
            # Calculate metrics
            accuracy = correct / total if total > 0 else 0.0
            avg_loss = total_loss / total if total > 0 else float('inf')
            
            # Calculate zero-day specific metrics
            zero_day_metrics = {}
            if zero_day_samples > 0:
                zero_day_detection_rate = zero_day_tp / zero_day_samples
                zero_day_metrics['zero_day_detection_rate'] = float(zero_day_detection_rate)
            else:
                zero_day_metrics['zero_day_detection_rate'] = 0.0
            
            if (zero_day_fp + zero_day_tn) > 0:
                zero_day_fp_rate = zero_day_fp / (zero_day_fp + zero_day_tn)
                zero_day_metrics['zero_day_fp_rate'] = float(zero_day_fp_rate)
            else:
                zero_day_metrics['zero_day_fp_rate'] = 0.0
            
            if (zero_day_tp + zero_day_fp + zero_day_fn + zero_day_tn) > 0:
                zero_day_accuracy = (zero_day_tp + zero_day_tn) / (zero_day_tp + zero_day_fp + zero_day_fn + zero_day_tn)
                zero_day_metrics['zero_day_accuracy'] = float(zero_day_accuracy)
            else:
                zero_day_metrics['zero_day_accuracy'] = 0.0
            
            zero_day_metrics['zero_day_samples'] = zero_day_samples
            zero_day_metrics['missing_attack'] = self.missing_attack
            
            logger.info(f"Client {self.client_id} evaluation complete:")
            logger.info(f"  Overall - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
            logger.info(f"  Zero-day ({self.missing_attack}) - Samples: {zero_day_samples}, "
                       f"Detection Rate: {zero_day_metrics['zero_day_detection_rate']:.4f}, "
                       f"FP Rate: {zero_day_metrics['zero_day_fp_rate']:.4f}")
            
            return float(avg_loss), len(self.test_loader.dataset), {
                "accuracy": float(accuracy),
                **zero_day_metrics
            }
            
        except Exception as e:
            logger.error(f"Client {self.client_id} evaluation failed: {e}")
            logger.error(traceback.format_exc())
            return float('inf'), 0, {"accuracy": 0.0}

def main():
    try:
        # Load and partition data for zero-day simulation
        logger.info(f"Loading zero-day simulation data for client {CLIENT_ID}")
        (X_train, y_train), (X_test, y_test), missing_attack = load_and_partition_data(
            file_path="Bot_IoT.csv",
            client_id=CLIENT_ID,
            num_clients=NUM_CLIENTS,
            label_col="category",
            chunk_size=146740
        )
        
        # Validate data
        if len(X_train) < 100:
            logger.warning(f"Client {CLIENT_ID} has very few training samples ({len(X_train)})")
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=min(BATCH_SIZE, len(train_dataset)), 
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=min(BATCH_SIZE, len(test_dataset)), 
            shuffle=False
        )
        
        logger.info(f"Data loaded - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
        logger.info(f"Missing attack type: {missing_attack}")
        
        # Log class distribution
        train_classes, train_counts = torch.unique(y_train, return_counts=True)
        test_classes, test_counts = torch.unique(y_test, return_counts=True)
        
        logger.info(f"Training class distribution: {dict(zip(train_classes.tolist(), train_counts.tolist()))}")
        logger.info(f"Testing class distribution: {dict(zip(test_classes.tolist(), test_counts.tolist()))}")
        
        # Create model
        num_features = X_train.shape[1]
        model = Net(input_size=num_features, output_size=NUM_GLOBAL_CLASSES)
        
        logger.info(f"Model created - Input: {num_features}, Output: {NUM_GLOBAL_CLASSES}")
        
        # Create Flower client
        client = ZeroDayFlowerClient(
            model, 
            train_loader, 
            test_loader, 
            DEVICE, 
            CLIENT_ID, 
            missing_attack
        )
        
        # Start client with retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Connecting to server (attempt {attempt + 1}/{max_retries})")
                fl.client.start_numpy_client(
                    server_address="localhost:8080", 
                    client=client
                )
                break
            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    raise e
        
    except Exception as e:
        logger.error(f"Client {CLIENT_ID} failed to start: {str(e)}")
        logger.error(traceback.format_exc())
        raise e

if __name__ == "__main__":
    main()