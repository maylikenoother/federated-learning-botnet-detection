import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl
import logging
from collections import OrderedDict
import traceback
import time
import pandas as pd

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

def get_global_class_info(csv_file, label_col="category"):
    """
    Get the global number of classes and class mapping by scanning the entire dataset.
    This ensures all clients use the same model architecture.
    """
    try:
        logger.info("🔍 Determining global class information...")
        
        # Read a sample to get all unique classes
        # We'll read in chunks to handle large files
        unique_classes = set()
        chunk_size = 100000
        
        for chunk in pd.read_csv(csv_file, chunksize=chunk_size, usecols=[label_col]):
            unique_classes.update(chunk[label_col].unique())
        
        # Sort classes for consistent ordering across all clients
        global_classes = sorted(list(unique_classes))
        num_global_classes = len(global_classes)
        
        logger.info(f"✅ Found {num_global_classes} global classes: {global_classes}")
        
        # Create class to index mapping
        class_to_idx = {cls: idx for idx, cls in enumerate(global_classes)}
        
        return num_global_classes, global_classes, class_to_idx
        
    except Exception as e:
        logger.error(f"Failed to determine global classes: {e}")
        # Fallback: assume common IoT attack categories
        default_classes = ['Normal', 'DDoS', 'DoS', 'Reconnaissance', 'Theft']
        logger.warning(f"Using default classes: {default_classes}")
        return len(default_classes), default_classes, {cls: idx for idx, cls in enumerate(default_classes)}

# Get global class information
NUM_GLOBAL_CLASSES, GLOBAL_CLASSES, CLASS_TO_IDX = get_global_class_info("Bot_IoT.csv", "category")

logger.info(f"🚀 Starting client {CLIENT_ID} on device: {DEVICE}")
logger.info(f"🎯 Using global model with {NUM_GLOBAL_CLASSES} classes")

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device, client_id, local_classes, class_mapping):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.client_id = client_id
        self.local_classes = local_classes
        self.class_mapping = class_mapping  # Maps local class indices to global class indices
        
        logger.info(f"Client {client_id} initialized - Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")
        logger.info(f"Client {client_id} local classes: {local_classes}")
        logger.info(f"Client {client_id} class mapping: {class_mapping}")

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
            logger.error(f"Model state dict keys: {list(self.model.state_dict().keys())}")
            logger.error(f"Parameter shapes: {[torch.tensor(p).shape for p in parameters]}")
            logger.error(f"Expected shapes: {[v.shape for v in self.model.state_dict().values()]}")
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
                    
                    # Map local class indices to global class indices
                    target_global = torch.zeros_like(target)
                    for local_idx, global_idx in self.class_mapping.items():
                        mask = target == local_idx
                        target_global[mask] = global_idx
                    
                    # Validate data
                    if torch.isnan(data).any() or torch.isinf(data).any():
                        logger.warning(f"Client {self.client_id}: Found NaN/Inf in training data, skipping batch")
                        continue
                    
                    # Check if target indices are valid
                    if target_global.max() >= NUM_GLOBAL_CLASSES or target_global.min() < 0:
                        logger.warning(f"Client {self.client_id}: Invalid target indices, skipping batch")
                        continue
                    
                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = criterion(output, target_global)
                    
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
                    
                    # Map local class indices to global class indices
                    target_global = torch.zeros_like(target)
                    for local_idx, global_idx in self.class_mapping.items():
                        mask = target == local_idx
                        target_global[mask] = global_idx
                    
                    # Validate data
                    if torch.isnan(data).any() or torch.isinf(data).any():
                        logger.warning(f"Client {self.client_id}: Found NaN/Inf in test data, skipping batch")
                        continue
                    
                    # Check if target indices are valid
                    if target_global.max() >= NUM_GLOBAL_CLASSES or target_global.min() < 0:
                        logger.warning(f"Client {self.client_id}: Invalid target indices in evaluation, skipping batch")
                        continue
                    
                    output = self.model(data)
                    loss = criterion(output, target_global)
                    
                    # Check for valid loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(f"Client {self.client_id}: Invalid loss in evaluation, skipping batch")
                        continue
                    
                    total_loss += loss.item() * data.size(0)  # Scale by batch size
                    
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target_global.view_as(pred)).sum().item()
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

def create_class_mapping(local_classes, global_classes):
    """Create mapping from local class indices to global class indices."""
    mapping = {}
    for local_idx, class_name in enumerate(local_classes):
        if class_name in global_classes:
            global_idx = global_classes.index(class_name)
            mapping[local_idx] = global_idx
        else:
            logger.warning(f"Local class '{class_name}' not found in global classes. Using index 0.")
            mapping[local_idx] = 0  # Default to first global class
    
    return mapping

def main():
    try:
        # Load and partition data
        logger.info(f"Loading data for client {CLIENT_ID}")
        X, y = load_and_partition_data(
            file_path="Bot_IoT.csv",
            client_id=CLIENT_ID,
            num_clients=NUM_CLIENTS,
            label_col="category",
            chunk_size=146740
        )
        
        # Get local classes from the loaded data
        local_class_indices = torch.unique(y).numpy()
        
        # We need to map back to class names to create proper mapping
        # For now, we'll assume the data loading preserves the original mapping
        # This is a simplification - in practice, you'd want to store the label encoder
        
        # Create a simple mapping assuming classes are consistently encoded
        local_classes = [GLOBAL_CLASSES[i] if i < len(GLOBAL_CLASSES) else f"class_{i}" 
                        for i in local_class_indices]
        
        logger.info(f"Client {CLIENT_ID} local classes: {local_classes}")
        
        # Create class mapping
        class_mapping = create_class_mapping([GLOBAL_CLASSES[i] for i in local_class_indices], GLOBAL_CLASSES)
        
        # Validate data
        if len(X) < 100:
            logger.warning(f"Client {CLIENT_ID} has very few samples ({len(X)}). Consider adjusting chunk_size.")
        
        # Split into train and test (80/20 split)
        dataset_size = len(X)
        train_size = max(int(0.8 * dataset_size), 1)
        test_size = dataset_size - train_size
        
        if test_size == 0:
            test_size = 1
            train_size = dataset_size - 1
        
        train_dataset = TensorDataset(X[:train_size], y[:train_size])
        test_dataset = TensorDataset(X[train_size:train_size + test_size], y[train_size:train_size + test_size])
        
        train_loader = DataLoader(train_dataset, batch_size=min(BATCH_SIZE, len(train_dataset)), shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=min(BATCH_SIZE, len(test_dataset)), shuffle=False)
        
        logger.info(f"Data loaded - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
        
        # Create model with GLOBAL number of classes
        num_features = X.shape[1]
        model = Net(input_size=num_features, output_size=NUM_GLOBAL_CLASSES)
        
        logger.info(f"Model created - Input: {num_features}, Output: {NUM_GLOBAL_CLASSES} (global classes)")
        
        # Create Flower client
        client = FlowerClient(model, train_loader, test_loader, DEVICE, CLIENT_ID, local_classes, class_mapping)
        
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