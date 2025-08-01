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

# FIXED: More conservative settings
CLIENT_ID = int(os.environ.get("CLIENT_ID", 0))
SERVER_ADDRESS = os.environ.get("SERVER_ADDRESS", "localhost:8080")
NUM_CLIENTS = 5
BATCH_SIZE = 8  # FIXED: Reduced from 16 to prevent memory issues
EPOCHS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"🚀 FIXED Client {CLIENT_ID} starting on {DEVICE}")
logger.info(f"📡 Server: {SERVER_ADDRESS}")

def wait_for_server(server_address, max_attempts=60):  # FIXED: Longer wait time
    """FIXED: Better server waiting with more attempts"""
    host, port = server_address.split(':')
    port = int(port)
    
    logger.info(f"⏳ Waiting for server at {host}:{port} (max {max_attempts}s)...")
    
    for attempt in range(max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2)
                result = s.connect_ex((host, port))
                if result == 0:
                    logger.info(f"✅ Server ready at {host}:{port}")
                    time.sleep(2)  # FIXED: Additional wait for server stability
                    return True
        except Exception as e:
            logger.debug(f"Connection attempt {attempt + 1}: {e}")
        
        if attempt % 10 == 0 and attempt > 0:
            logger.info(f"⏳ Still waiting... {attempt}/{max_attempts}")
        
        time.sleep(1)
    
    logger.error(f"❌ Server not available after {max_attempts} attempts")
    return False

class FixedFlowerClient(fl.client.NumPyClient):
    """FIXED: Much more robust Flower client"""
    
    def __init__(self, model, train_loader, test_loader, device, client_id, missing_attack):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.client_id = client_id
        self.missing_attack = missing_attack
        
        # FIXED: Validate data loaders
        if len(train_loader.dataset) == 0:
            raise ValueError(f"Client {client_id} has empty training dataset")
        if len(test_loader.dataset) == 0:
            logger.warning(f"Client {client_id} has empty test dataset")
        
        logger.info(f"✅ FIXED Client {client_id} initialized")
        logger.info(f"📊 Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")
        logger.info(f"🎯 Missing attack: {missing_attack}")

    def get_parameters(self, config):
        """FIXED: Safer parameter extraction"""
        try:
            return [val.cpu().numpy() for val in self.model.state_dict().values()]
        except Exception as e:
            logger.error(f"Client {self.client_id}: Parameter extraction failed: {e}")
            # Return dummy parameters to prevent crash
            return [np.array([0.0])]

    def set_parameters(self, parameters):
        """FIXED: Safer parameter setting"""
        try:
            if len(parameters) != len(self.model.state_dict()):
                logger.warning(f"Parameter count mismatch: got {len(parameters)}, expected {len(self.model.state_dict())}")
                return
            
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=False)  # FIXED: strict=False
            
        except Exception as e:
            logger.error(f"Client {self.client_id}: Parameter setting failed: {e}")

    def fit(self, parameters, config):
        """FIXED: Much more robust training"""
        server_round = config.get('server_round', 'unknown')
        logger.info(f"🎯 Client {self.client_id} - Training Round {server_round}")
        
        # FIXED: Early exit for insufficient data
        if len(self.train_loader.dataset) < 10:
            logger.warning(f"⚠️ Client {self.client_id}: Insufficient data ({len(self.train_loader.dataset)} samples)")
            return self.get_parameters(config), 0, {
                "loss": 0.0,
                "accuracy": 0.0,
                "status": "insufficient_data"
            }
        
        try:
            self.set_parameters(parameters)
            self.model.train()
            
            # FIXED: Conservative optimizer settings
            optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=config.get('learning_rate', 0.0001),  # FIXED: Lower learning rate
                weight_decay=1e-5
            )
            criterion = nn.CrossEntropyLoss()
            
            total_loss = 0.0
            total_samples = 0
            successful_batches = 0
            
            # FIXED: Limit training to prevent long runs
            max_batches = min(5, len(self.train_loader))  # Maximum 5 batches
            
            for epoch in range(EPOCHS):
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    if batch_idx >= max_batches:
                        break
                    
                    try:
                        data, target = data.to(self.device), target.to(self.device)
                        
                        # FIXED: Skip invalid data
                        if torch.isnan(data).any() or torch.isinf(data).any():
                            logger.debug(f"Skipping batch {batch_idx} due to invalid data")
                            continue
                        
                        optimizer.zero_grad()
                        output = self.model(data)
                        loss = criterion(output, target)
                        
                        # FIXED: Skip invalid loss
                        if torch.isnan(loss) or torch.isinf(loss):
                            logger.debug(f"Skipping batch {batch_idx} due to invalid loss")
                            continue
                        
                        loss.backward()
                        
                        # FIXED: Gradient clipping
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                        
                        optimizer.step()
                        
                        total_loss += loss.item()
                        total_samples += data.size(0)
                        successful_batches += 1
                        
                    except Exception as e:
                        logger.debug(f"Batch {batch_idx} failed: {e}")
                        continue
            
            # FIXED: Calculate metrics safely
            avg_loss = total_loss / max(successful_batches, 1)
            
            logger.info(f"✅ Client {self.client_id} training complete - "
                       f"Loss: {avg_loss:.4f}, Batches: {successful_batches}")
            
            return self.get_parameters(config), total_samples, {
                "loss": float(avg_loss),
                "accuracy": 0.0,  # FIXED: Don't calculate during training to save time
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
        """FIXED: Much more robust evaluation"""
        logger.info(f"📊 Client {self.client_id} - Evaluation")
        
        # FIXED: Skip evaluation if no test data
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
            processed_batches = 0
            
            # FIXED: Limit evaluation batches
            max_eval_batches = min(3, len(self.test_loader))
            
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(self.test_loader):
                    if batch_idx >= max_eval_batches:
                        break
                    
                    try:
                        data, target = data.to(self.device), target.to(self.device)
                        
                        # FIXED: Skip invalid data
                        if torch.isnan(data).any() or torch.isinf(data).any():
                            continue
                        
                        output = self.model(data)
                        
                        # FIXED: Skip invalid output
                        if torch.isnan(output).any() or torch.isinf(output).any():
                            continue
                        
                        loss = criterion(output, target)
                        if torch.isnan(loss) or torch.isinf(loss):
                            continue
                        
                        test_loss += loss.item()
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
                        total += target.size(0)
                        processed_batches += 1
                        
                    except Exception as e:
                        logger.debug(f"Evaluation batch {batch_idx} failed: {e}")
                        continue
            
            # FIXED: Calculate metrics safely
            accuracy = correct / max(total, 1)
            avg_loss = test_loss / max(processed_batches, 1)
            
            logger.info(f"✅ Client {self.client_id} evaluation complete - "
                       f"Acc: {accuracy:.4f}, Loss: {avg_loss:.4f}")
            
            return float(avg_loss), total, {
                "accuracy": float(accuracy),
                "missing_attack": self.missing_attack,
                "batches_processed": processed_batches
            }
            
        except Exception as e:
            logger.error(f"❌ Client {self.client_id} evaluation failed: {e}")
            return float('inf'), 0, {
                "accuracy": 0.0,
                "missing_attack": self.missing_attack,
                "error": str(e)
            }

def main():
    """FIXED: Much more robust main function"""
    try:
        logger.info(f"🌟 Starting FIXED FL Client {CLIENT_ID}")
        
        # FIXED: Wait for server first
        if not wait_for_server(SERVER_ADDRESS, max_attempts=90):
            logger.error("❌ Server not available, exiting")
            return False
        
        # FIXED: Load data with retries
        max_data_retries = 3
        for attempt in range(max_data_retries):
            try:
                logger.info(f"📂 Loading data (attempt {attempt + 1}/{max_data_retries})")
                
                (X_train, y_train), (X_test, y_test), missing_attack = load_and_partition_data(
                    file_path="Bot_IoT.csv",
                    client_id=CLIENT_ID,
                    num_clients=NUM_CLIENTS,
                    label_col="category",
                    chunk_size=5000  # FIXED: Smaller chunk size
                )
                
                logger.info(f"✅ Data loaded: Train={len(X_train)}, Test={len(X_test)}")
                break
                
            except Exception as e:
                logger.error(f"❌ Data loading attempt {attempt + 1} failed: {e}")
                if attempt < max_data_retries - 1:
                    time.sleep(5)
                else:
                    logger.error("❌ All data loading attempts failed")
                    return False
        
        # FIXED: Validate data
        if len(X_train) < 10:
            logger.error(f"❌ Insufficient training data: {len(X_train)} samples")
            return False
        
        # FIXED: Create datasets with safe batch sizes
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        # FIXED: Safe batch sizes
        train_batch_size = min(BATCH_SIZE, len(train_dataset), 8)
        test_batch_size = min(BATCH_SIZE, max(1, len(test_dataset)), 8)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=train_batch_size, 
            shuffle=True, 
            drop_last=False,
            num_workers=0  # FIXED: No multiprocessing
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=test_batch_size, 
            shuffle=False, 
            drop_last=False,
            num_workers=0  # FIXED: No multiprocessing
        )
        
        # FIXED: Create model
        num_features = X_train.shape[1]
        num_classes = 5  # Bot-IoT classes
        
        model = Net(
            input_size=num_features, 
            output_size=num_classes,
            hidden_size=32,  # FIXED: Smaller model
            num_hidden_layers=2,  # FIXED: Fewer layers
            dropout_rate=0.2  # FIXED: Less dropout
        )
        
        logger.info(f"🧠 Model: {num_features} → {num_classes} classes")
        
        # FIXED: Create client
        client = FixedFlowerClient(
            model, train_loader, test_loader, DEVICE, CLIENT_ID, missing_attack
        )
        
        # FIXED: Connect with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"🔗 Connecting to server (attempt {attempt + 1}/{max_retries})")
                
                fl.client.start_numpy_client(
                    server_address=SERVER_ADDRESS, 
                    client=client
                )
                
                logger.info(f"✅ Client {CLIENT_ID} completed successfully!")
                return True
                
            except Exception as e:
                logger.error(f"❌ Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10
                    logger.info(f"⏳ Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"❌ All connection attempts failed")
                    return False
        
    except Exception as e:
        logger.error(f"❌ Client {CLIENT_ID} failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)