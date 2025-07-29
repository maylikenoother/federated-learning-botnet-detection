import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import logging
import pickle
import os
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global configurations for zero-day simulation
GLOBAL_LABEL_ENCODER = None
GLOBAL_CLASSES = None
GLOBAL_FEATURE_COLUMNS = None
ZERO_DAY_CONFIG = {
    0: "DDoS",        # Client 0 missing DDoS attacks
    1: "Reconnaissance", # Client 1 missing Reconnaissance attacks  
    2: "Theft",       # Client 2 missing Theft attacks
    3: "DoS",         # Client 3 missing DoS attacks
    4: "Normal"       # Client 4 missing Normal traffic
}

def get_or_create_global_encoder(file_path, label_col="category"):
    """
    Create or load a global label encoder and determine consistent feature columns.
    """
    global GLOBAL_LABEL_ENCODER, GLOBAL_CLASSES, GLOBAL_FEATURE_COLUMNS
    
    encoder_file = "global_label_encoder.pkl"
    classes_file = "global_classes.pkl"
    features_file = "global_features.pkl"
    
    # Try to load existing encoders
    if (os.path.exists(encoder_file) and os.path.exists(classes_file) and 
        os.path.exists(features_file)):
        try:
            with open(encoder_file, 'rb') as f:
                GLOBAL_LABEL_ENCODER = pickle.load(f)
            with open(classes_file, 'rb') as f:
                GLOBAL_CLASSES = pickle.load(f)
            with open(features_file, 'rb') as f:
                GLOBAL_FEATURE_COLUMNS = pickle.load(f)
            logger.info(f"✅ Loaded global encoder with {len(GLOBAL_CLASSES)} classes: {GLOBAL_CLASSES}")
            logger.info(f"✅ Loaded global features: {len(GLOBAL_FEATURE_COLUMNS)} columns")
            return GLOBAL_LABEL_ENCODER, GLOBAL_CLASSES, GLOBAL_FEATURE_COLUMNS
        except Exception as e:
            logger.warning(f"Failed to load existing encoders: {e}. Creating new ones.")
    
    # Create new global encoder and feature set
    logger.info("🔍 Scanning dataset to create global encoders...")
    
    try:
        # Read a representative sample to determine feature columns and classes
        sample_df = pd.read_csv(file_path, nrows=50000, low_memory=False)
        sample_df = sample_df.dropna()
        
        # Get unique classes
        unique_classes = sample_df[label_col].dropna().unique()
        GLOBAL_CLASSES = sorted(list(unique_classes))
        
        # Create label encoder
        GLOBAL_LABEL_ENCODER = LabelEncoder()
        GLOBAL_LABEL_ENCODER.fit(GLOBAL_CLASSES)
        
        # Determine consistent feature columns (numeric only)
        feature_df = sample_df.drop(columns=[label_col])
        numeric_columns = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove zero variance columns
        valid_columns = []
        for col in numeric_columns:
            if sample_df[col].std() > 1e-8:
                valid_columns.append(col)
        
        GLOBAL_FEATURE_COLUMNS = sorted(valid_columns)
        
        logger.info(f"✅ Created global encoder with {len(GLOBAL_CLASSES)} classes: {GLOBAL_CLASSES}")
        logger.info(f"✅ Determined {len(GLOBAL_FEATURE_COLUMNS)} consistent feature columns")
        
        # Save encoders
        with open(encoder_file, 'wb') as f:
            pickle.dump(GLOBAL_LABEL_ENCODER, f)
        with open(classes_file, 'wb') as f:
            pickle.dump(GLOBAL_CLASSES, f)
        with open(features_file, 'wb') as f:
            pickle.dump(GLOBAL_FEATURE_COLUMNS, f)
        
    except Exception as e:
        logger.error(f"Failed to create global encoders: {e}")
        # Fallback
        GLOBAL_CLASSES = ['DDoS', 'DoS', 'Normal', 'Reconnaissance', 'Theft']
        GLOBAL_LABEL_ENCODER = LabelEncoder()
        GLOBAL_LABEL_ENCODER.fit(GLOBAL_CLASSES)
        GLOBAL_FEATURE_COLUMNS = [f'feature_{i}' for i in range(30)]  # Fallback features
        logger.warning(f"Using fallback configuration")
    
    return GLOBAL_LABEL_ENCODER, GLOBAL_CLASSES, GLOBAL_FEATURE_COLUMNS

def load_stratified_data_sample(file_path, label_col="category", samples_per_class=5000):
    """
    Load a stratified sample of data ensuring all classes are represented.
    """
    logger.info(f"🔍 Loading stratified sample with {samples_per_class} samples per class")
    
    class_data = {cls: [] for cls in GLOBAL_CLASSES}
    chunk_size = 100000
    total_processed = 0
    
    try:
        for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
            chunk = chunk.dropna()
            total_processed += len(chunk)
            
            # Collect samples for each class
            for class_name in GLOBAL_CLASSES:
                if len(class_data[class_name]) >= samples_per_class:
                    continue
                    
                class_samples = chunk[chunk[label_col] == class_name]
                
                if len(class_samples) > 0:
                    needed = samples_per_class - len(class_data[class_name])
                    to_take = min(needed, len(class_samples))
                    
                    sampled = class_samples.sample(n=to_take, random_state=42)
                    class_data[class_name].append(sampled)
                    
                    current_total = sum(len(df) for df in class_data[class_name])
                    logger.info(f"  {class_name}: collected {to_take}, total: {current_total}")
            
            # Check if we have enough data for all classes
            if all(sum(len(df) for df in class_data[cls]) >= samples_per_class 
                  for cls in GLOBAL_CLASSES):
                logger.info("✅ Collected sufficient samples for all classes")
                break
                
            if total_processed >= 1000000:  # Process up to 1M rows
                logger.info(f"Processed {total_processed} rows, stopping search")
                break
    
    except Exception as e:
        logger.error(f"Error loading stratified sample: {e}")
        raise e
    
    # Combine data for each class
    combined_data = []
    final_distribution = {}
    
    for class_name, data_list in class_data.items():
        if data_list:
            class_df = pd.concat(data_list, ignore_index=True)
            # Take exactly the requested number of samples
            if len(class_df) > samples_per_class:
                class_df = class_df.sample(n=samples_per_class, random_state=42)
            combined_data.append(class_df)
            final_distribution[class_name] = len(class_df)
            logger.info(f"✅ {class_name}: {len(class_df)} samples")
        else:
            logger.warning(f"⚠️  No samples found for {class_name}")
    
    if not combined_data:
        raise ValueError("No data collected for any class")
    
    # Combine all data
    full_dataset = pd.concat(combined_data, ignore_index=True)
    full_dataset = full_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"✅ Total stratified sample: {len(full_dataset)} rows")
    
    return full_dataset, final_distribution

def create_client_partition(full_dataset, client_id, missing_attack, label_col="category"):
    """
    Create client-specific partition from the full dataset.
    """
    logger.info(f"🎯 Creating partition for client {client_id}, excluding {missing_attack}")
    
    # Remove the missing attack type
    available_data = full_dataset[full_dataset[label_col] != missing_attack].copy()
    
    # Define client-specific sampling preferences (non-IID)
    client_preferences = {
        0: {'DoS': 0.6, 'Normal': 0.2, 'Reconnaissance': 0.15, 'Theft': 0.05},
        1: {'DDoS': 0.4, 'DoS': 0.3, 'Normal': 0.25, 'Theft': 0.05},
        2: {'DDoS': 0.45, 'DoS': 0.3, 'Normal': 0.15, 'Reconnaissance': 0.1},
        3: {'DDoS': 0.5, 'Normal': 0.25, 'Reconnaissance': 0.2, 'Theft': 0.05},
        4: {'DDoS': 0.4, 'DoS': 0.35, 'Reconnaissance': 0.15, 'Theft': 0.1}
    }
    
    preferences = client_preferences.get(client_id, {})
    target_total = 25000  # Target samples per client
    
    client_parts = []
    
    for class_name, proportion in preferences.items():
        if class_name == missing_attack:
            continue
            
        class_data = available_data[available_data[label_col] == class_name]
        target_samples = int(target_total * proportion)
        
        if len(class_data) > 0:
            actual_samples = min(target_samples, len(class_data))
            sampled_data = class_data.sample(n=actual_samples, random_state=42 + client_id)
            client_parts.append(sampled_data)
            logger.info(f"  Client {client_id} - {class_name}: {len(sampled_data)} samples")
    
    if not client_parts:
        raise ValueError(f"No training data for client {client_id}")
    
    # Combine client data
    client_data = pd.concat(client_parts, ignore_index=True)
    client_data = client_data.sample(frac=1, random_state=42 + client_id).reset_index(drop=True)
    
    logger.info(f"✅ Client {client_id} total training samples: {len(client_data)}")
    
    return client_data

def create_test_data(full_dataset, client_id, label_col="category"):
    """
    Create test data that includes ALL classes for zero-day evaluation.
    """
    logger.info(f"🧪 Creating test data for client {client_id}")
    
    test_parts = []
    test_samples_per_class = 800  # Smaller test sets
    
    for class_name in GLOBAL_CLASSES:
        class_data = full_dataset[full_dataset[label_col] == class_name]
        
        if len(class_data) > 0:
            # Use different random seed for test data
            actual_samples = min(test_samples_per_class, len(class_data))
            sampled_test = class_data.sample(n=actual_samples, random_state=100 + client_id)
            test_parts.append(sampled_test)
            
            missing_attack = ZERO_DAY_CONFIG.get(client_id, "DDoS")
            zero_day_marker = " (ZERO-DAY)" if class_name == missing_attack else ""
            logger.info(f"  Test {class_name}: {len(sampled_test)} samples{zero_day_marker}")
    
    if not test_parts:
        raise ValueError(f"No test data for client {client_id}")
    
    test_data = pd.concat(test_parts, ignore_index=True)
    test_data = test_data.sample(frac=1, random_state=100 + client_id).reset_index(drop=True)
    
    logger.info(f"✅ Client {client_id} total test samples: {len(test_data)}")
    
    return test_data

def load_and_partition_data(file_path, client_id, num_clients, label_col="category", chunk_size=None):
    """
    Load and partition data implementing zero-day attack simulation for federated learning.
    Uses consistent feature extraction to avoid dimension mismatches.
    """
    logger.info(f"📦 Loading zero-day simulation data for client {client_id}/{num_clients}")

    try:
        # Get or create global encoders and feature columns
        global_encoder, global_classes, global_features = get_or_create_global_encoder(file_path, label_col)
        
        # Get the attack type this client won't see during training
        missing_attack = ZERO_DAY_CONFIG.get(client_id, "DDoS")
        logger.info(f"Client {client_id} will NOT see '{missing_attack}' attacks during training")
        
        # Load stratified sample of the entire dataset
        full_dataset, distribution = load_stratified_data_sample(file_path, label_col, samples_per_class=8000)
        
        # Create client-specific training partition (excluding missing attack)
        train_data = create_client_partition(full_dataset, client_id, missing_attack, label_col)
        
        # Create test data (includes ALL attack types)
        test_data = create_test_data(full_dataset, client_id, label_col)
        
        # Process data with consistent features
        X_train, y_train = process_features_and_labels(
            train_data, label_col, global_encoder, global_classes, global_features, "training"
        )
        
        X_test, y_test = process_features_and_labels(
            test_data, label_col, global_encoder, global_classes, global_features, "testing"
        )
        
        # Validate zero-day scenario
        train_classes = set(train_data[label_col].unique())
        test_classes = set(test_data[label_col].unique())
        zero_day_classes = test_classes - train_classes
        
        logger.info(f"✅ Zero-day simulation created for client {client_id}")
        logger.info(f"   Training classes: {sorted(train_classes)}")
        logger.info(f"   Test classes: {sorted(test_classes)}")
        logger.info(f"   Zero-day classes: {sorted(zero_day_classes)}")
        logger.info(f"   Feature dimensions: Train={X_train.shape}, Test={X_test.shape}")
        
        return (X_train, y_train), (X_test, y_test), missing_attack
        
    except Exception as e:
        logger.error(f"Failed to load data for client {client_id}: {str(e)}")
        raise e

def process_features_and_labels(df, label_col, global_encoder, global_classes, global_features, phase):
    """
    Process features and labels with consistent feature columns across all clients.
    """
    # Separate features and labels
    y_raw = df[label_col].values.astype(str)
    X_df = df.drop(columns=[label_col])
    
    # Use only the globally consistent feature columns
    available_features = [col for col in global_features if col in X_df.columns]
    
    if len(available_features) == 0:
        logger.warning("No matching feature columns found, using all numeric columns")
        available_features = X_df.select_dtypes(include=[np.number]).columns.tolist()
    
    X = X_df[available_features].values
    
    logger.info(f"✅ {phase} features: {X.shape[1]} columns ({len(available_features)} available)")
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Add small amount of noise to prevent overfitting
    noise_scale = 0.005
    X += np.random.normal(0, noise_scale, X.shape)
    
    # Encode labels using global encoder
    y_encoded = []
    unknown_count = 0
    
    for label in y_raw:
        try:
            encoded_label = global_encoder.transform([label])[0]
            y_encoded.append(encoded_label)
        except ValueError:
            logger.warning(f"Unknown class encountered: {label}")
            encoded_label = 0  # Map to first class temporarily
            y_encoded.append(encoded_label)
            unknown_count += 1
    
    y = np.array(y_encoded)
    
    if unknown_count > 0:
        logger.warning(f"Found {unknown_count} unknown class labels in {phase} data")
    
    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    # Final validation
    if torch.isnan(X_tensor).any() or torch.isinf(X_tensor).any():
        logger.warning("Found NaN/Inf in features, cleaning...")
        X_tensor = torch.nan_to_num(X_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
    
    max_label = y_tensor.max().item()
    min_label = y_tensor.min().item()
    
    if max_label >= len(global_classes) or min_label < 0:
        logger.error(f"Invalid label range: {min_label} to {max_label}, num_classes: {len(global_classes)}")
        raise ValueError("Label indices out of range")
    
    logger.info(f"✅ {phase} data processed: X={X_tensor.shape}, y={y_tensor.shape}")
    
    return X_tensor, y_tensor

def validate_zero_day_setup(file_path, num_clients=5, label_col="category"):
    """
    Validate the zero-day setup across all clients.
    """
    logger.info("🔍 Validating zero-day federated learning setup...")
    
    # First create global encoders
    get_or_create_global_encoder(file_path, label_col)
    
    client_info = {}
    
    for client_id in range(num_clients):
        try:
            (X_train, y_train), (X_test, y_test), missing_attack = load_and_partition_data(
                file_path=file_path,
                client_id=client_id,
                num_clients=num_clients,
                label_col=label_col
            )
            
            train_classes = torch.unique(y_train).numpy()
            test_classes = torch.unique(y_test).numpy()
            
            client_info[client_id] = {
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'train_features': X_train.shape[1],
                'test_features': X_test.shape[1],
                'train_classes': train_classes,
                'test_classes': test_classes,
                'missing_attack': missing_attack,
                'zero_day_present': len(test_classes) > len(train_classes)
            }
            
            logger.info(f"✅ Client {client_id}: {len(X_train)} train, {len(X_test)} test samples")
            logger.info(f"   Features: {X_train.shape[1]} (train), {X_test.shape[1]} (test)")
            logger.info(f"   Missing attack: {missing_attack}")
            logger.info(f"   Zero-day scenario: {client_info[client_id]['zero_day_present']}")
            
        except Exception as e:
            logger.error(f"❌ Client {client_id} validation failed: {e}")
    
    # Summary
    total_train = sum(info['train_samples'] for info in client_info.values())
    total_test = sum(info['test_samples'] for info in client_info.values())
    
    logger.info(f"\n📊 ZERO-DAY SETUP VALIDATION COMPLETE")
    logger.info(f"Total training samples: {total_train:,}")
    logger.info(f"Total test samples: {total_test:,}")
    logger.info(f"Zero-day scenarios: {sum(1 for info in client_info.values() if info['zero_day_present'])}/{num_clients}")
    
    # Check feature consistency
    feature_counts = [info['train_features'] for info in client_info.values()]
    if len(set(feature_counts)) == 1:
        logger.info(f"✅ All clients have consistent feature dimensions: {feature_counts[0]}")
    else:
        logger.warning(f"⚠️  Inconsistent feature dimensions: {feature_counts}")
    
    return client_info

if __name__ == "__main__":
    # Validate the zero-day setup
    validate_zero_day_setup(
        file_path="Bot_IoT.csv",
        num_clients=5,
        label_col="category"
    )