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

# FIXED: Better logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FIXED: More realistic settings
CLIENT_ID = int(os.environ.get("CLIENT_ID", 0))
SERVER_ADDRESS = os.environ.get("SERVER_ADDRESS", "localhost:8080")
NUM_CLIENTS = 5
BATCH_SIZE = 32  # Increased for better training
EPOCHS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"🚀 FIXED Client {CLIENT_ID} starting on {DEVICE}")
logger.info(f"📡 Server: {SERVER_ADDRESS}")

def wait_for_server(server_address, max_attempts=60):
    """Wait for server to be ready"""
    host, port = server_address.split(':')
    port = int(port)
    
    logger.info(f"⏳ Waiting for server at {host}:{port}...")
    
    for attempt in range(max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2)
                result = s.connect_ex((host, port))
                if result == 0:
                    logger.info(f"✅ Server ready at {host}:{port}")
                    time.sleep(2)
                    return True
        except Exception as e:
            logger.debug(f"Connection attempt {attempt + 1}: {e}")
        
        if attempt % 10 == 0 and attempt > 0:
            logger.info(f"⏳ Still waiting... {attempt}/{max_attempts}")
        
        time.sleep(1)
    
    logger.error(f"❌ Server not available after {max_attempts} attempts")
    return False

class FixedFlowerClient(fl.client.NumPyClient):
    """FIXED: Flower client with proper zero-day detection"""
    
    def __init__(self, model, train_loader, test_loader, device, client_id, missing_attack):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.client_id = client_id
        self.missing_attack = missing_attack
        
        # Create label mapping for zero-day detection
        self.attack_types = ['Normal', 'DDoS', 'DoS', 'Reconnaissance', 'Theft']
        self.missing_attack_id = self.attack_types.index(missing_attack) if missing_attack in self.attack_types else 0
        
        # Validate data loaders
        if len(train_loader.dataset) == 0:
            raise ValueError(f"Client {client_id} has empty training dataset")
        if len(test_loader.dataset) == 0:
            logger.warning(f"Client {client_id} has empty test dataset")
        
        logger.info(f"✅ FIXED Client {client_id} initialized")
        logger.info(f"📊 Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")
        logger.info(f"🎯 Missing attack: {missing_attack} (ID: {self.missing_attack_id})")

    def get_parameters(self, config):
        """Get model parameters"""
        try:
            return [val.cpu().numpy() for val in self.model.state_dict().values()]
        except Exception as e:
            logger.error(f"Client {self.client_id}: Parameter extraction failed: {e}")
            return [np.array([0.0])]

    def set_parameters(self, parameters):
        """Set model parameters"""
        try:
            if len(parameters) != len(self.model.state_dict()):
                logger.warning(f"Parameter count mismatch: got {len(parameters)}, expected {len(self.model.state_dict())}")
                return
            
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=False)
            
        except Exception as e:
            logger.error(f"Client {self.client_id}: Parameter setting failed: {e}")

    def fit(self, parameters, config):
        """FIXED: Enhanced training with realistic batch processing"""
        server_round = config.get('server_round', 'unknown')
        logger.info(f"🎯 Client {self.client_id} - Training Round {server_round}")
        
        # Check for sufficient data
        if len(self.train_loader.dataset) < 50:
            logger.warning(f"⚠️ Client {self.client_id}: Insufficient data ({len(self.train_loader.dataset)} samples)")
            return self.get_parameters(config), 0, {
                "loss": 0.0,
                "accuracy": 0.0,
                "status": "insufficient_data"
            }
        
        try:
            self.set_parameters(parameters)
            self.model.train()
            
            # FIXED: Better optimizer settings
            optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=config.get('learning_rate', 0.001),
                weight_decay=1e-4
            )
            criterion = nn.CrossEntropyLoss()
            
            total_loss = 0.0
            total_samples = 0
            successful_batches = 0
            correct_predictions = 0
            
            # Process all batches (or reasonable subset)
            max_batches = min(50, len(self.train_loader))  # Process up to 50 batches
            
            for epoch in range(EPOCHS):
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    if batch_idx >= max_batches:
                        break
                    
                    try:
                        data, target = data.to(self.device), target.to(self.device)
                        
                        # Skip invalid data
                        if torch.isnan(data).any() or torch.isinf(data).any():
                            logger.debug(f"Skipping batch {batch_idx} due to invalid data")
                            continue
                        
                        optimizer.zero_grad()
                        output = self.model(data)
                        loss = criterion(output, target)
                        
                        # Skip invalid loss
                        if torch.isnan(loss) or torch.isinf(loss):
                            logger.debug(f"Skipping batch {batch_idx} due to invalid loss")
                            continue
                        
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                        # Calculate metrics
                        total_loss += loss.item()
                        total_samples += data.size(0)
                        successful_batches += 1
                        
                        # Calculate training accuracy
                        pred = output.argmax(dim=1, keepdim=True)
                        correct_predictions += pred.eq(target.view_as(pred)).sum().item()
                        
                    except Exception as e:
                        logger.debug(f"Batch {batch_idx} failed: {e}")
                        continue
            
            # Calculate final metrics
            avg_loss = total_loss / max(successful_batches, 1)
            accuracy = correct_predictions / max(total_samples, 1)
            
            logger.info(f"✅ Client {self.client_id} training complete - "
                       f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Batches: {successful_batches}")
            
            return self.get_parameters(config), total_samples, {
                "loss": float(avg_loss),
                "accuracy": float(accuracy),
                "batches_processed": successful_batches,
                "missing_attack": self.missing_attack
            }
            
        except Exception as e:
            logger.error(f"❌ Client {self.client_id} training failed: {e}")
            return self.get_parameters(config), 0, {
                "loss": float('inf'),
                "accuracy": 0.0,
                "error": str(e),
                "missing_attack": self.missing_attack
            }

    def evaluate(self, parameters, config):
        """FIXED: Enhanced evaluation with proper zero-day detection metrics"""
        logger.info(f"📊 Client {self.client_id} - Evaluation with Zero-Day Detection")
        
        # Skip evaluation if no test data
        if len(self.test_loader.dataset) == 0:
            logger.warning(f"⚠️ Client {self.client_id}: No test data")
            return 0.0, 0, {
                "accuracy": 0.0,
                "missing_attack": self.missing_attack,
                "status": "no_test_data"
            }
        
        try:
            self.set_parameters(parameters)
            self.model.eval()
            
            criterion = nn.CrossEntropyLoss()
            test_loss = 0.0
            correct = 0
            total = 0
            
            # FIXED: Zero-day detection metrics
            zero_day_tp = 0  # True positives for missing attack
            zero_day_fp = 0  # False positives for missing attack  
            zero_day_tn = 0  # True negatives for missing attack
            zero_day_fn = 0  # False negatives for missing attack
            zero_day_total = 0  # Total missing attack samples
            
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(self.test_loader):
                    try:
                        data, target = data.to(self.device), target.to(self.device)
                        
                        # Skip invalid data
                        if torch.isnan(data).any() or torch.isinf(data).any():
                            continue
                        
                        output = self.model(data)
                        
                        # Skip invalid output
                        if torch.isnan(output).any() or torch.isinf(output).any():
                            continue
                        
                        loss = criterion(output, target)
                        if torch.isnan(loss) or torch.isinf(loss):
                            continue
                        
                        test_loss += loss.item()
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
                        total += target.size(0)
                        
                        # FIXED: Calculate zero-day detection metrics
                        for i in range(len(target)):
                            true_label = target[i].item()
                            pred_label = pred[i].item()
                            
                            if true_label == self.missing_attack_id:
                                # This is a zero-day attack sample
                                zero_day_total += 1
                                if pred_label == self.missing_attack_id:
                                    zero_day_tp += 1  # Correctly detected zero-day
                                else:
                                    zero_day_fn += 1  # Missed zero-day attack
                            else:
                                # This is not a zero-day attack
                                if pred_label == self.missing_attack_id:
                                    zero_day_fp += 1  # False alarm
                                else:
                                    zero_day_tn += 1  # Correctly identified as not zero-day
                        
                    except Exception as e:
                        logger.debug(f"Evaluation batch {batch_idx} failed: {e}")
                        continue
            
            # Calculate overall metrics
            accuracy = correct / max(total, 1)
            avg_loss = test_loss / max(total, 1)
            
            # FIXED: Calculate zero-day detection metrics
            zero_day_precision = zero_day_tp / max(zero_day_tp + zero_day_fp, 1)
            zero_day_recall = zero_day_tp / max(zero_day_tp + zero_day_fn, 1)
            zero_day_f1 = 2 * (zero_day_precision * zero_day_recall) / max(zero_day_precision + zero_day_recall, 1)
            zero_day_detection_rate = zero_day_tp / max(zero_day_total, 1)
            
            # False positive rate for zero-day
            zero_day_fpr = zero_day_fp / max(zero_day_fp + zero_day_tn, 1)
            
            logger.info(f"✅ Client {self.client_id} evaluation complete:")
            logger.info(f"   Overall Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}")
            logger.info(f"   Zero-day samples: {zero_day_total}")
            logger.info(f"   Zero-day detection rate: {zero_day_detection_rate:.4f}")
            logger.info(f"   Zero-day precision: {zero_day_precision:.4f}")
            logger.info(f"   Zero-day recall: {zero_day_recall:.4f}")
            logger.info(f"   Zero-day F1-score: {zero_day_f1:.4f}")
            
            return float(avg_loss), total, {
                "accuracy": float(accuracy),
                "missing_attack": self.missing_attack,
                "zero_day_detection_rate": float(zero_day_detection_rate),
                "zero_day_precision": float(zero_day_precision),
                "zero_day_recall": float(zero_day_recall),
                "zero_day_f1_score": float(zero_day_f1),
                "zero_day_false_positive_rate": float(zero_day_fpr),
                "zero_day_samples": zero_day_total,
                "total_samples": total
            }
            
        except Exception as e:
            logger.error(f"❌ Client {self.client_id} evaluation failed: {e}")
            return float('inf'), 0, {
                "accuracy": 0.0,
                "missing_attack": self.missing_attack,
                "error": str(e)
            }

def main():
    """FIXED: Main function with realistic data loading"""
    try:
        logger.info(f"🌟 Starting FIXED FL Client {CLIENT_ID}")
        
        # Wait for server first
        if not wait_for_server(SERVER_ADDRESS, max_attempts=90):
            logger.error("❌ Server not available, exiting")
            return False
        
        # FIXED: Load data with realistic chunk size
        logger.info(f"📂 Loading realistic dataset for client {CLIENT_ID}")
        
        (X_train, y_train), (X_test, y_test), missing_attack = load_and_partition_data(
            file_path="Bot_IoT.csv",
            client_id=CLIENT_ID,
            num_clients=NUM_CLIENTS,
            label_col="category",
            chunk_size=50000  # Much larger chunk for realistic training
        )
        
        logger.info(f"✅ Data loaded: Train={len(X_train)}, Test={len(X_test)}")
        
        # Validate data sizes
        if len(X_train) < 100:
            logger.error(f"❌ Insufficient training data: {len(X_train)} samples")
            return False
        
        # Create datasets with realistic batch sizes
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        # FIXED: Realistic batch sizes
        train_batch_size = min(BATCH_SIZE, len(train_dataset))
        test_batch_size = min(BATCH_SIZE, max(1, len(test_dataset)))
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=train_batch_size, 
            shuffle=True, 
            drop_last=False,
            num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=test_batch_size, 
            shuffle=False, 
            drop_last=False,
            num_workers=0
        )
        
        # Create model with proper input size
        num_features = X_train.shape[1]
        num_classes = 5  # Bot-IoT classes
        
        model = Net(
            input_size=num_features, 
            output_size=num_classes,
            hidden_size=64,  # Better model capacity
            num_hidden_layers=3,
            dropout_rate=0.3
        )
        
        logger.info(f"🧠 Model: {num_features} → {num_classes} classes")
        
        # Create client
        client = FixedFlowerClient(
            model, train_loader, test_loader, DEVICE, CLIENT_ID, missing_attack
        )
        
        # Connect to server
        logger.info(f"🔗 Connecting to server at {SERVER_ADDRESS}")
        
        fl.client.start_numpy_client(
            server_address=SERVER_ADDRESS, 
            client=client
        )
        
        logger.info(f"✅ Client {CLIENT_ID} completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Client {CLIENT_ID} failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)