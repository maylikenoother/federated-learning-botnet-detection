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

from model import Net, train_model, test_model, calculate_zero_day_metrics
from partition_data import load_and_partition_data, ZERO_DAY_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CLIENT_ID = int(os.environ.get("CLIENT_ID", 0))
NUM_CLIENTS = 5
BATCH_SIZE = 32  # Reduced batch size for more stable training
EPOCHS = 2  # Increased epochs for better learning
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global class information - we'll get this from the partition module
GLOBAL_CLASSES = ['DDoS', 'DoS', 'Normal', 'Reconnaissance', 'Theft']

logger.info(f"🚀 Starting zero-day simulation client {CLIENT_ID} on device: {DEVICE}")

class ZeroDayFlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device, client_id, missing_attack, global_classes):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.client_id = client_id
        self.missing_attack = missing_attack
        self.global_classes = global_classes
        self.missing_attack_class_id = None
        
        # Find the class ID of the missing attack
        if missing_attack in global_classes:
            self.missing_attack_class_id = global_classes.index(missing_attack)
        
        logger.info(f"🎯 Client {client_id} initialized for zero-day simulation")
        logger.info(f"   Train samples: {len(train_loader.dataset)}")
        logger.info(f"   Test samples: {len(test_loader.dataset)}")
        logger.info(f"   Missing attack: {missing_attack} (class_id: {self.missing_attack_class_id})")
        
        # Log training data distribution
        self._log_data_distribution()

    def _log_data_distribution(self):
        """Log the distribution of classes in training and test data."""
        train_labels = []
        test_labels = []
        
        # Get training labels
        for _, labels in self.train_loader:
            train_labels.extend(labels.numpy())
        
        # Get test labels
        for _, labels in self.test_loader:
            test_labels.extend(labels.numpy())
        
        # Count distributions
        train_counts = np.bincount(train_labels, minlength=len(self.global_classes))
        test_counts = np.bincount(test_labels, minlength=len(self.global_classes))
        
        logger.info(f"Client {self.client_id} data distribution:")
        for i, class_name in enumerate(self.global_classes):
            zero_day_marker = " (ZERO-DAY)" if i == self.missing_attack_class_id else ""
            logger.info(f"   {class_name}: Train={train_counts[i]}, Test={test_counts[i]}{zero_day_marker}")

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
        """Train the model with local data using zero-day simulation."""
        server_round = config.get('server_round', 'unknown')
        logger.info(f"🎯 Client {self.client_id} - Training Round {server_round} (Zero-Day Simulation)")
        
        try:
            # Update local model parameters
            self.set_parameters(parameters)
            
            # Train model with enhanced training function
            learning_rate = config.get('learning_rate', 0.0005)  # Lower learning rate for stability
            local_epochs = config.get('local_epochs', EPOCHS)
            
            avg_loss, accuracy = train_model(
                model=self.model,
                train_loader=self.train_loader,
                device=self.device,
                epochs=local_epochs,
                learning_rate=learning_rate,
                use_focal_loss=True  # Use focal loss for imbalanced data
            )
            
            logger.info(f"Client {self.client_id} training completed:")
            logger.info(f"   Average loss: {avg_loss:.6f}")
            logger.info(f"   Training accuracy: {accuracy:.4f}")
            logger.info(f"   Missing attack type: {self.missing_attack}")
            
            # Return training metrics
            return (
                self.get_parameters(config={}), 
                len(self.train_loader.dataset), 
                {
                    "loss": float(avg_loss),
                    "accuracy": float(accuracy),
                    "missing_attack": self.missing_attack,
                    "client_id": self.client_id
                }
            )
            
        except Exception as e:
            logger.error(f"Client {self.client_id} training failed: {e}")
            logger.error(traceback.format_exc())
            raise e

    def evaluate(self, parameters, config):
        """Evaluate the model on local data including zero-day attack detection."""
        logger.info(f"📊 Client {self.client_id} - Evaluating (Zero-Day Detection)")
        
        try:
            # Update local model parameters
            self.set_parameters(parameters)
            
            # Evaluate model with detailed metrics
            results = test_model(
                model=self.model,
                test_loader=self.test_loader,
                device=self.device,
                detailed_metrics=True
            )
            
            avg_loss = results['loss']
            accuracy = results['accuracy']
            total_samples = results['total_samples']
            
            # Calculate zero-day specific metrics
            zero_day_metrics = {}
            if 'predictions' in results and 'targets' in results and self.missing_attack_class_id is not None:
                zero_day_metrics = calculate_zero_day_metrics(
                    predictions=results['predictions'],
                    targets=results['targets'],
                    missing_attack_class_id=self.missing_attack_class_id
                )
            
            # Log detailed results
            logger.info(f"Client {self.client_id} evaluation complete:")
            logger.info(f"   Overall accuracy: {accuracy:.4f}")
            logger.info(f"   Overall loss: {avg_loss:.6f}")
            logger.info(f"   Total test samples: {total_samples}")
            
            if zero_day_metrics:
                logger.info(f"   Zero-day detection accuracy: {zero_day_metrics.get('zero_day_accuracy', 0):.4f}")
                logger.info(f"   Zero-day false positive rate: {zero_day_metrics.get('zero_day_fp_rate', 0):.4f}")
                logger.info(f"   Zero-day samples tested: {zero_day_metrics.get('zero_day_samples', 0)}")
            
            # Log per-class accuracies if available
            if 'class_accuracies' in results:
                logger.info("   Per-class accuracies:")
                for class_id, class_acc in results['class_accuracies'].items():
                    class_name = self.global_classes[class_id] if class_id < len(self.global_classes) else f"class_{class_id}"
                    zero_day_marker = " (ZERO-DAY)" if class_id == self.missing_attack_class_id else ""
                    logger.info(f"      {class_name}: {class_acc:.4f}{zero_day_marker}")
            
            # Prepare return metrics
            evaluation_metrics = {
                "accuracy": float(accuracy),
                "loss": float(avg_loss),
                "missing_attack": self.missing_attack,
                "client_id": self.client_id,
                "total_samples": total_samples
            }
            
            # Add zero-day metrics
            evaluation_metrics.update(zero_day_metrics)
            
            return float(avg_loss), total_samples, evaluation_metrics
            
        except Exception as e:
            logger.error(f"Client {self.client_id} evaluation failed: {e}")
            logger.error(traceback.format_exc())
            # Return default values instead of crashing
            return float('inf'), 0, {
                "accuracy": 0.0,
                "missing_attack": self.missing_attack,
                "client_id": self.client_id,
                "error": str(e)
            }

def validate_data_quality(train_data, test_data, missing_attack):
    """Validate that the data quality is appropriate for zero-day simulation."""
    train_features, train_labels = train_data
    test_features, test_labels = test_data
    
    # Check basic data quality
    if len(train_features) == 0 or len(test_features) == 0:
        raise ValueError("Empty training or test data")
    
    if train_features.shape[1] == 0:
        raise ValueError("No features in data")
    
    # Check class distribution
    train_classes = torch.unique(train_labels)
    test_classes = torch.unique(test_labels)
    
    logger.info(f"Data validation:")
    logger.info(f"   Training samples: {len(train_features)}")
    logger.info(f"   Test samples: {len(test_features)}")
    logger.info(f"   Features: {train_features.shape[1]}")
    logger.info(f"   Training classes: {len(train_classes)}")
    logger.info(f"   Test classes: {len(test_classes)}")
    
    # Check for zero-day scenario
    zero_day_classes = set(test_classes.numpy()) - set(train_classes.numpy())
    if len(zero_day_classes) > 0:
        logger.info(f"   ✅ Zero-day classes in test: {zero_day_classes}")
    else:
        logger.warning(f"   ⚠️  No zero-day classes detected!")
    
    # Check for minimum samples per class
    if len(train_classes) > 0:
        train_class_counts = torch.bincount(train_labels)
        min_samples = train_class_counts.min().item()
        if min_samples < 10:
            logger.warning(f"   ⚠️  Some classes have very few samples (min: {min_samples})")
    
    return True

def main():
    try:
        # Load and partition data for zero-day simulation
        logger.info(f"Loading zero-day simulation data for client {CLIENT_ID}")
        
        # CORRECTED: Handle the new return format with 3 values
        (X_train, y_train), (X_test, y_test), missing_attack = load_and_partition_data(
            file_path="Bot_IoT.csv",
            client_id=CLIENT_ID,
            num_clients=NUM_CLIENTS,
            label_col="category",
            chunk_size=80000  # Smaller chunks for more controlled experiments
        )
        
        # Validate data quality
        validate_data_quality((X_train, y_train), (X_test, y_test), missing_attack)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        # Use smaller batch sizes for more stable training
        train_batch_size = min(BATCH_SIZE, len(train_dataset))
        test_batch_size = min(BATCH_SIZE, len(test_dataset))
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=train_batch_size, 
            shuffle=True,
            drop_last=False  # Don't drop last batch to avoid losing data
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=test_batch_size, 
            shuffle=False,
            drop_last=False
        )
        
        logger.info(f"Data loaders created:")
        logger.info(f"   Train batches: {len(train_loader)}")
        logger.info(f"   Test batches: {len(test_loader)}")
        
        # Create model with appropriate architecture
        num_features = X_train.shape[1]
        num_classes = len(GLOBAL_CLASSES)
        
        # Use the optimal architecture from Popoola et al.
        model = Net(
            input_size=num_features,
            output_size=num_classes,
            hidden_size=100,
            num_hidden_layers=4,
            dropout_rate=0.3  # Add dropout for regularization
        )
        
        logger.info(f"Model created:")
        logger.info(f"   Input features: {num_features}")
        logger.info(f"   Output classes: {num_classes}")
        logger.info(f"   Architecture: 4-layer DNN with 100 hidden units")
        logger.info(f"   Global classes: {GLOBAL_CLASSES}")
        
        # Create Flower client
        client = ZeroDayFlowerClient(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=DEVICE,
            client_id=CLIENT_ID,
            missing_attack=missing_attack,
            global_classes=GLOBAL_CLASSES
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
                    time.sleep(5)  # Wait before retry
                else:
                    raise e
        
    except Exception as e:
        logger.error(f"Client {CLIENT_ID} failed to start: {str(e)}")
        logger.error(traceback.format_exc())
        raise e

if __name__ == "__main__":
    main()