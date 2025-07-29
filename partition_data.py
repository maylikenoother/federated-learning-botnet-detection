import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import logging
import pickle
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global label encoder to ensure consistent class encoding across all clients
GLOBAL_LABEL_ENCODER = None
GLOBAL_CLASSES = None

def get_or_create_global_encoder(file_path, label_col="category"):
    """
    Create or load a global label encoder that ensures consistent class encoding
    across all clients by scanning the entire dataset once.
    """
    global GLOBAL_LABEL_ENCODER, GLOBAL_CLASSES
    
    encoder_file = "global_label_encoder.pkl"
    classes_file = "global_classes.pkl"
    
    # Try to load existing encoder
    if os.path.exists(encoder_file) and os.path.exists(classes_file):
        try:
            with open(encoder_file, 'rb') as f:
                GLOBAL_LABEL_ENCODER = pickle.load(f)
            with open(classes_file, 'rb') as f:
                GLOBAL_CLASSES = pickle.load(f)
            logger.info(f"✅ Loaded global encoder with {len(GLOBAL_CLASSES)} classes: {GLOBAL_CLASSES}")
            return GLOBAL_LABEL_ENCODER, GLOBAL_CLASSES
        except Exception as e:
            logger.warning(f"Failed to load existing encoder: {e}. Creating new one.")
    
    # Create new global encoder by scanning entire dataset
    logger.info("🔍 Scanning entire dataset to create global class encoder...")
    
    unique_classes = set()
    chunk_size = 100000
    total_rows = 0
    
    try:
        for chunk in pd.read_csv(file_path, chunksize=chunk_size, usecols=[label_col]):
            chunk_classes = chunk[label_col].dropna().unique()
            unique_classes.update(chunk_classes)
            total_rows += len(chunk)
            
            if total_rows % 500000 == 0:
                logger.info(f"Processed {total_rows} rows, found {len(unique_classes)} unique classes")
        
        # Sort classes for consistent ordering
        GLOBAL_CLASSES = sorted(list(unique_classes))
        
        # Create label encoder
        GLOBAL_LABEL_ENCODER = LabelEncoder()
        GLOBAL_LABEL_ENCODER.fit(GLOBAL_CLASSES)
        
        logger.info(f"✅ Created global encoder with {len(GLOBAL_CLASSES)} classes: {GLOBAL_CLASSES}")
        
        # Save encoder for future use
        with open(encoder_file, 'wb') as f:
            pickle.dump(GLOBAL_LABEL_ENCODER, f)
        with open(classes_file, 'wb') as f:
            pickle.dump(GLOBAL_CLASSES, f)
        
        logger.info(f"💾 Saved global encoder to {encoder_file}")
        
    except Exception as e:
        logger.error(f"Failed to create global encoder: {e}")
        # Fallback to common IoT classes
        GLOBAL_CLASSES = ['DDoS', 'DoS', 'Normal', 'Reconnaissance', 'Theft']
        GLOBAL_LABEL_ENCODER = LabelEncoder()
        GLOBAL_LABEL_ENCODER.fit(GLOBAL_CLASSES)
        logger.warning(f"Using fallback classes: {GLOBAL_CLASSES}")
    
    return GLOBAL_LABEL_ENCODER, GLOBAL_CLASSES

def load_and_partition_data(file_path, client_id, num_clients, label_col="label", chunk_size=100_000):
    """
    Load and partition data for federated learning clients with consistent global class encoding.
    """
    logger.info(f"📦 Loading data for client {client_id}/{num_clients}")

    try:
        # Get or create global encoder
        global_encoder, global_classes = get_or_create_global_encoder(file_path, label_col)
        
        # Calculate which chunk this client should get
        start_row = client_id * chunk_size
        end_row = start_row + chunk_size
        
        logger.info(f"Client {client_id} will load rows {start_row} to {end_row}")
        
        # Read the specific chunk for this client
        df = pd.read_csv(
            file_path, 
            skiprows=range(1, start_row + 1) if start_row > 0 else None,
            nrows=chunk_size,
            low_memory=False, 
            skipinitialspace=True
        )
        
        logger.info(f"✅ Loaded {len(df)} rows for client {client_id}")
        
        # Clean the data
        initial_rows = len(df)
        df.dropna(inplace=True)
        logger.info(f"Dropped {initial_rows - len(df)} rows with missing values")
        
        if len(df) == 0:
            raise ValueError(f"No data remaining after cleaning for client {client_id}")
        
        # Check if label column exists
        if label_col not in df.columns:
            logger.error(f"Available columns: {df.columns.tolist()}")
            raise ValueError(f"Label column '{label_col}' not found in DataFrame.")
        
        # Separate features and labels
        y_raw = df[label_col].values.astype(str)
        X_df = df.drop(columns=[label_col])
        
        # Select only numeric columns for features
        numeric_columns = X_df.select_dtypes(include=[np.number]).columns
        X = X_df[numeric_columns].values
        
        logger.info(f"✅ Data shape: X={X.shape}, y={y_raw.shape}")
        logger.info(f"Features used: {len(numeric_columns)} numeric columns")
        
        # Get unique classes in this client's data
        local_unique_classes = np.unique(y_raw)
        logger.info(f"Unique labels in client data: {local_unique_classes}")
        
        # Check for valid data
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(f"No valid features found for client {client_id}")
        
        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        logger.info("✅ Features normalized")
        
        # Encode labels using GLOBAL encoder
        # Handle unknown classes by mapping them to a default class
        y_encoded = []
        unknown_class_count = 0
        
        for label in y_raw:
            try:
                encoded_label = global_encoder.transform([label])[0]
                y_encoded.append(encoded_label)
            except ValueError:
                # Unknown class - map to first class (index 0)
                y_encoded.append(0)
                unknown_class_count += 1
        
        y = np.array(y_encoded)
        
        if unknown_class_count > 0:
            logger.warning(f"Found {unknown_class_count} unknown class labels, mapped to class 0")
        
        logger.info(f"✅ Labels encoded using global encoder: {len(global_classes)} total classes")
        logger.info(f"Local data has classes with indices: {np.unique(y)}")
        
        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        # Validate tensors
        if torch.isnan(X_tensor).any() or torch.isinf(X_tensor).any():
            logger.warning("Found NaN or Inf values in features, replacing with zeros")
            X_tensor = torch.nan_to_num(X_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Validate label indices
        max_label = y_tensor.max().item()
        min_label = y_tensor.min().item()
        
        if max_label >= len(global_classes) or min_label < 0:
            logger.error(f"Invalid label indices: min={min_label}, max={max_label}, num_classes={len(global_classes)}")
            raise ValueError(f"Label indices out of range for client {client_id}")
        
        logger.info(f"✅ Data preparation complete for client {client_id}")
        logger.info(f"Final tensor shapes: X={X_tensor.shape}, y={y_tensor.shape}")
        logger.info(f"Label range: {min_label} to {max_label} (total global classes: {len(global_classes)})")
        
        return X_tensor, y_tensor
        
    except Exception as e:
        logger.error(f"Failed to load data for client {client_id}: {str(e)}")
        raise e

def get_global_class_info():
    """Get global class information after encoder is created"""
    global GLOBAL_LABEL_ENCODER, GLOBAL_CLASSES
    
    if GLOBAL_CLASSES is None or GLOBAL_LABEL_ENCODER is None:
        # Try to load from files
        try:
            with open("global_classes.pkl", 'rb') as f:
                GLOBAL_CLASSES = pickle.load(f)
            with open("global_label_encoder.pkl", 'rb') as f:
                GLOBAL_LABEL_ENCODER = pickle.load(f)
        except:
            # Fallback
            GLOBAL_CLASSES = ['DDoS', 'DoS', 'Normal', 'Reconnaissance', 'Theft']
            GLOBAL_LABEL_ENCODER = LabelEncoder()
            GLOBAL_LABEL_ENCODER.fit(GLOBAL_CLASSES)
    
    return len(GLOBAL_CLASSES), GLOBAL_CLASSES, GLOBAL_LABEL_ENCODER

def validate_data_distribution(file_path, num_clients, chunk_size, label_col="category"):
    """
    Validate that data is properly distributed across clients with consistent encoding.
    """
    logger.info("🔍 Validating data distribution across clients...")
    
    # First, create global encoder
    get_or_create_global_encoder(file_path, label_col)
    
    total_samples = 0
    all_classes = set()
    
    for client_id in range(num_clients):
        try:
            X, y = load_and_partition_data(
                file_path=file_path,
                client_id=client_id,
                num_clients=num_clients,
                label_col=label_col,
                chunk_size=chunk_size
            )
            
            client_classes = torch.unique(y).numpy()
            all_classes.update(client_classes)
            total_samples += len(X)
            
            logger.info(f"Client {client_id}: {X.shape[0]} samples, classes {client_classes}")
            
        except Exception as e:
            logger.error(f"Client {client_id} failed: {str(e)}")
    
    logger.info(f"✅ Total samples across all clients: {total_samples}")
    logger.info(f"✅ All class indices found: {sorted(all_classes)}")
    logger.info("✅ Data distribution validation complete")

if __name__ == "__main__":
    # Test the data loading
    validate_data_distribution(
        file_path="Bot_IoT.csv",
        num_clients=5,
        chunk_size=146740,
        label_col="category"
    )