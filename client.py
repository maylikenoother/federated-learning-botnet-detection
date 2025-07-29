import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl
import logging
from collections import OrderedDict
import traceback
import time

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

logger.info(f"🚀 Starting client {CLIENT_ID} on device: {DEVICE}")

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device, client_id):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.client_id = client_id
        logger.info(f"Client {client_id} initialized - Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")

    def get_parameters(self, config):
        """Return the model parameters as a list of NumPy ndarrays."""
        try:
            parameters = [val.cpu().numpy() for val in self.model.state_dict().values()]
            logger.debug(f"Client {self.client_id}: Retrieved {len(parameters)} parameter arrays")
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
            logger.debug(f"Client {self.client_id}: Updated model parameters")
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
                    # Move data to device
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # Validate data
                    if torch.isnan(data).any() or torch.isinf(data).any():
                        logger.warning(f"Client {self.client_id}: Found NaN/Inf in training data, skipping batch")
                        continue
                    
                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = criterion(output, target)
                    
                    # Check for valid loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(f"Client {self.client_id}: Invalid loss detected, skipping batch")
                        continue
                    
                    loss.backward()
                    
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    epoch_samples += data.size(0)
                
                total_loss += epoch_loss
                total_samples += epoch_samples
                
                if epoch_samples > 0:
                    avg_epoch_loss = epoch_loss / epoch_samples
                    logger.info(f"Client {self.client_id} - Epoch {epoch+1}/{EPOCHS}, Loss: {avg_epoch_loss:.4f}")
            
            # Calculate average loss per sample
            if total_samples > 0:
                avg_loss = total_loss / total_samples
            else:
                avg_loss = float('inf')
                logger.warning(f"Client {self.client_id}: No samples processed during training!")
            
            logger.info(f"Client {self.client_id} training completed - Average loss: {avg_loss:.4f}")
            
            return self.get_parameters(config={}), len(self.train_loader.dataset), {"loss": float(avg_loss)}
            
        except Exception as e:
            logger.error(f"Client {self.client_id} training failed: {e}")
            logger.error(traceback.format_exc())
            raise e

    def evaluate(self, parameters, config):
        """Evaluate the model on local data."""
        logger.info(f"📊 Client {self.client_id} - Evaluating")
        
        try:
            # Update local model parameters
            self.set_parameters(parameters)
            
            # Evaluate model
            self.model.eval()
            criterion = nn.CrossEntropyLoss()
            
            total_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(self.test_loader):
                    # Move data to device
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # Validate data
                    if torch.isnan(data).any() or torch.isinf(data).any():
                        logger.warning(f"Client {self.client_id}: Found NaN/Inf in test data, skipping batch")
                        continue
                    
                    output = self.model(data)
                    loss = criterion(output, target)
                    
                    # Check for valid loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(f"Client {self.client_id}: Invalid loss in evaluation, skipping batch")
                        continue
                    
                    total_loss += loss.item() * data.size(0)  # Scale by batch size
                    
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
            
            # Calculate metrics
            if total > 0:
                accuracy = correct / total
                avg_loss = total_loss / total
            else:
                accuracy = 0.0
                avg_loss = float('inf')
                logger.warning(f"Client {self.client_id}: No samples processed during evaluation!")
            
            logger.info(f"Client {self.client_id} evaluation complete - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
            
            return float(avg_loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}
            
        except Exception as e:
            logger.error(f"Client {self.client_id} evaluation failed: {e}")
            logger.error(traceback.format_exc())
            # Return default values instead of crashing
            return float('inf'), 0, {"accuracy": 0.0}

def main():
    try:
        # Load and partition data
        logger.info(f"Loading data for client {CLIENT_ID}")
        X, y = load_and_partition_data(
            file_path="Bot_IoT.csv",
            client_id=CLIENT_ID,
            num_clients=NUM_CLIENTS,
            label_col="category",
            chunk_size=146740  # Reduced chunk size for more reliable loading
        )
        
        # Validate data
        if len(X) < 100:
            logger.warning(f"Client {CLIENT_ID} has very few samples ({len(X)}). Consider adjusting chunk_size.")
        
        # Split into train and test (80/20 split)
        dataset_size = len(X)
        train_size = max(int(0.8 * dataset_size), 1)  # Ensure at least 1 training sample
        test_size = dataset_size - train_size
        
        if test_size == 0:
            test_size = 1
            train_size = dataset_size - 1
        
        train_dataset = TensorDataset(X[:train_size], y[:train_size])
        test_dataset = TensorDataset(X[train_size:train_size + test_size], y[train_size:train_size + test_size])
        
        train_loader = DataLoader(train_dataset, batch_size=min(BATCH_SIZE, len(train_dataset)), shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=min(BATCH_SIZE, len(test_dataset)), shuffle=False)
        
        logger.info(f"Data loaded - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
        
        # Create model
        num_features = X.shape[1]
        num_classes = len(torch.unique(y))
        model = Net(input_size=num_features, output_size=num_classes)
        
        logger.info(f"Model created - Input: {num_features}, Output: {num_classes}")
        
        # Create Flower client
        client = FlowerClient(model, train_loader, test_loader, DEVICE, CLIENT_ID)
        
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
                    time.sleep(5)  # Wait before retry
                else:
                    raise e
        
    except Exception as e:
        logger.error(f"Client {CLIENT_ID} failed to start: {str(e)}")
        logger.error(traceback.format_exc())
        raise e

if __name__ == "__main__":
    main()