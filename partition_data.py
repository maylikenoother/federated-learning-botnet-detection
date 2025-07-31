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

# Attack type mapping for Bot-IoT dataset
ATTACK_MAPPING = {
    'Normal': 'Normal',
    'DDoS': 'DDoS', 
    'DoS': 'DoS',
    'Reconnaissance': 'Reconnaissance',
    'Theft': 'Theft'
}

# Client-specific missing attacks (as per the paper)
CLIENT_MISSING_ATTACKS = {
    0: 'DDoS',           # ED1: No DDoS samples
    1: 'Reconnaissance', # ED2: No Reconnaissance samples  
    2: 'Theft',         # ED3: No Theft samples
    3: 'DoS',           # ED4: No DoS samples
    4: 'Normal'         # ED5: No Normal samples
}

def load_global_encoder_and_features():
    """Load or create global encoder and feature list"""
    encoder_file = "global_label_encoder.pkl"
    features_file = "global_features.pkl"
    
    # For Bot-IoT, we know the classes
    global_classes = ['DDoS', 'DoS', 'Normal', 'Reconnaissance', 'Theft']
    
    # Create and save encoder
    global_encoder = LabelEncoder()
    global_encoder.fit(global_classes)
    
    with open(encoder_file, 'wb') as f:
        pickle.dump(global_encoder, f)
    
    logger.info(f"✅ Created global encoder with {len(global_classes)} classes: {global_classes}")
    
    # Define features (37 features as mentioned in the paper)
    # These are the features after removing redundant ones
    global_features = None  # Will be determined from the first data load
    
    return global_encoder, global_classes, global_features

def load_full_dataset(file_path, label_col="category", sample_size_per_class=8000):
    """
    Load the full dataset with balanced sampling across all classes
    """
    logger.info(f"🔍 Loading full dataset with balanced sampling")
    
    # First pass: determine all unique classes and their counts
    class_counts = {}
    chunk_size = 50000
    
    for chunk in pd.read_csv(file_path, chunksize=chunk_size, usecols=[label_col]):
        value_counts = chunk[label_col].value_counts()
        for class_name, count in value_counts.items():
            class_counts[class_name] = class_counts.get(class_name, 0) + count
    
    logger.info(f"Class distribution in dataset: {class_counts}")
    
    # Second pass: collect balanced samples
    samples_per_class = {cls: [] for cls in class_counts.keys()}
    samples_collected = {cls: 0 for cls in class_counts.keys()}
    
    # Calculate sampling probability for each class
    sampling_probs = {}
    for cls in class_counts.keys():
        # Ensure we don't try to sample more than available
        target_samples = min(sample_size_per_class, class_counts[cls])
        sampling_probs[cls] = target_samples / class_counts[cls]
    
    # Read and sample data
    all_data = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        for class_name in samples_per_class.keys():
            if samples_collected[class_name] < sample_size_per_class:
                class_data = chunk[chunk[label_col] == class_name]
                if len(class_data) > 0:
                    # Sample based on probability
                    n_to_sample = int(len(class_data) * sampling_probs[class_name])
                    n_to_sample = min(n_to_sample, sample_size_per_class - samples_collected[class_name])
                    
                    if n_to_sample > 0:
                        sampled = class_data.sample(n=min(n_to_sample, len(class_data)))
                        samples_per_class[class_name].append(sampled)
                        samples_collected[class_name] += len(sampled)
    
    # Combine all samples
    for class_name, samples in samples_per_class.items():
        if samples:
            all_data.extend(samples)
    
    if not all_data:
        raise ValueError("No data collected!")
    
    full_dataset = pd.concat(all_data, ignore_index=True)
    
    # Log final distribution
    final_distribution = full_dataset[label_col].value_counts()
    logger.info(f"✅ Loaded balanced dataset with distribution: {final_distribution.to_dict()}")
    
    return full_dataset

def create_zero_day_partition(full_dataset, client_id, missing_attack, label_col="category", 
                            train_ratio=0.8, min_samples_per_class=100):
    """
    Create training data for a client, excluding the missing attack type.
    Ensure balanced representation of remaining classes.
    """
    # First, add a unique identifier to track samples
    full_dataset = full_dataset.copy()
    full_dataset['_sample_id'] = range(len(full_dataset))
    
    # Get data excluding the missing attack for training
    train_data = full_dataset[full_dataset[label_col] != missing_attack].copy()
    
    if len(train_data) == 0:
        raise ValueError(f"No training data available for client {client_id} after excluding {missing_attack}")
    
    # Get remaining classes
    remaining_classes = train_data[label_col].unique()
    logger.info(f"Client {client_id} training classes: {remaining_classes}")
    
    # Balance the remaining classes
    balanced_data = []
    min_class_size = train_data[label_col].value_counts().min()
    
    # Sample equally from each remaining class
    samples_per_class = max(min_samples_per_class, min(1500, min_class_size))  # Reduced from 2000
    
    train_sample_ids = set()
    for cls in remaining_classes:
        class_data = train_data[train_data[label_col] == cls]
        if len(class_data) >= samples_per_class:
            sampled = class_data.sample(n=samples_per_class, random_state=42 + client_id)
        else:
            sampled = class_data
        balanced_data.append(sampled)
        train_sample_ids.update(sampled['_sample_id'].values)
    
    balanced_train_data = pd.concat(balanced_data, ignore_index=True)
    
    # Create test data (includes all classes including missing attack)
    test_samples_per_class = 200  # As per the paper, test set is smaller
    test_data_parts = []
    
    for cls in full_dataset[label_col].unique():
        # Get all samples of this class that are NOT in training set
        class_data = full_dataset[
            (full_dataset[label_col] == cls) & 
            (~full_dataset['_sample_id'].isin(train_sample_ids))
        ]
        
        # For the missing attack class, we can use all available samples (up to limit)
        # For other classes, we use samples not in training
        if len(class_data) >= test_samples_per_class:
            sampled = class_data.sample(n=test_samples_per_class, random_state=42 + client_id + 1000)
        else:
            # If we don't have enough samples, use what we have
            sampled = class_data
            # If still not enough, sample some from the full dataset for this class
            if len(sampled) < 50 and cls == missing_attack:  # Ensure minimum samples for zero-day class
                additional_samples = full_dataset[full_dataset[label_col] == cls].sample(
                    n=min(50, len(full_dataset[full_dataset[label_col] == cls])), 
                    random_state=42 + client_id + 2000
                )
                sampled = pd.concat([sampled, additional_samples]).drop_duplicates(subset=['_sample_id'])
        
        if len(sampled) > 0:
            test_data_parts.append(sampled)
    
    test_data = pd.concat(test_data_parts, ignore_index=True)
    
    # Remove the temporary sample_id column
    balanced_train_data = balanced_train_data.drop(columns=['_sample_id'])
    test_data = test_data.drop(columns=['_sample_id'])
    
    # Log distribution
    train_dist = balanced_train_data[label_col].value_counts()
    test_dist = test_data[label_col].value_counts()
    logger.info(f"Client {client_id} train distribution: {train_dist.to_dict()}")
    logger.info(f"Client {client_id} test distribution: {test_dist.to_dict()}")
    
    # Validate test set has samples
    if len(test_data) == 0:
        logger.error(f"❌ Client {client_id} has empty test set!")
        # Create minimal test set from training data as fallback
        test_data = balanced_train_data.sample(n=min(100, len(balanced_train_data)), random_state=42 + client_id + 3000)
        logger.warning(f"⚠️ Created fallback test set with {len(test_data)} samples")
    
    return balanced_train_data, test_data

def prepare_features_and_labels(data, label_col="category", global_encoder=None, feature_cols=None):
    """
    Prepare features and labels for training/testing
    """
    # Check if data is empty
    if len(data) == 0:
        logger.error("❌ Empty dataset provided to prepare_features_and_labels")
        # Return minimal tensors to prevent crashes
        return torch.zeros((1, 10)), torch.zeros(1, dtype=torch.long), []
    
    # Get labels
    y_raw = data[label_col].values
    
    # Select numeric features
    if feature_cols is None:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != label_col]
    
    if len(feature_cols) == 0:
        logger.error("❌ No numeric features found!")
        # Return minimal tensors
        return torch.zeros((len(data), 10)), torch.zeros(len(data), dtype=torch.long), []
    
    X = data[feature_cols].values
    
    # Handle missing values
    X = np.nan_to_num(X, nan=0.0)
    
    # Normalize features
    scaler = StandardScaler()
    # Handle case where all values are the same
    try:
        X = scaler.fit_transform(X)
    except:
        logger.warning("⚠️ StandardScaler failed, using raw values")
        X = X.astype(np.float32)
    
    # Encode labels
    if global_encoder is None:
        raise ValueError("Global encoder is required")
    
    # Handle unknown labels gracefully
    y = []
    for label in y_raw:
        try:
            encoded = global_encoder.transform([label])[0]
            y.append(encoded)
        except:
            logger.warning(f"⚠️ Unknown label '{label}', using class 0")
            y.append(0)
    
    y = np.array(y)
    
    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    return X_tensor, y_tensor, feature_cols

def load_and_partition_data(file_path, client_id, num_clients, label_col="category", chunk_size=100000):
    """
    Load and partition data for zero-day attack detection simulation
    """
    logger.info(f"📦 Loading zero-day simulation data for client {client_id}/{num_clients}")
    
    try:
        # Load global encoder
        global_encoder, global_classes, _ = load_global_encoder_and_features()
        
        # Get missing attack for this client
        missing_attack = CLIENT_MISSING_ATTACKS.get(client_id)
        if missing_attack is None:
            raise ValueError(f"No missing attack defined for client {client_id}")
        
        logger.info(f"Client {client_id} will NOT see '{missing_attack}' attacks during training")
        
        # Load full balanced dataset
        full_dataset = load_full_dataset(file_path, label_col, sample_size_per_class=8000)
        
        # Create zero-day partitions
        train_data, test_data = create_zero_day_partition(
            full_dataset, client_id, missing_attack, label_col
        )
        
        # Get feature columns from the first preparation
        X_train, y_train, feature_cols = prepare_features_and_labels(
            train_data, label_col, global_encoder
        )
        
        # Use same feature columns for test data
        X_test, y_test, _ = prepare_features_and_labels(
            test_data, label_col, global_encoder, feature_cols
        )
        
        logger.info(f"✅ Client {client_id} data prepared:")
        logger.info(f"   Training: {X_train.shape[0]} samples, {len(torch.unique(y_train))} classes")
        logger.info(f"   Testing: {X_test.shape[0]} samples, {len(torch.unique(y_test))} classes")
        logger.info(f"   Missing attack: {missing_attack}")
        
        # Final validation
        if len(X_test) == 0:
            logger.error(f"❌ Client {client_id} has no test samples!")
            # Create minimal test set from train data
            n_test = min(50, len(X_train))
            X_test = X_train[:n_test]
            y_test = y_train[:n_test]
            logger.warning(f"⚠️ Created fallback test set with {n_test} samples")
        
        return (X_train, y_train), (X_test, y_test), missing_attack
        
    except Exception as e:
        logger.error(f"Failed to load data for client {client_id}: {str(e)}")
        raise e

def validate_zero_day_setup(file_path, num_clients=5, label_col="category"):
    """
    Validate the zero-day attack detection setup
    """
    logger.info("🔍 Validating zero-day attack detection setup...")
    
    for client_id in range(num_clients):
        try:
            (X_train, y_train), (X_test, y_test), missing_attack = load_and_partition_data(
                file_path=file_path,
                client_id=client_id,
                num_clients=num_clients,
                label_col=label_col
            )
            
            logger.info(f"✅ Client {client_id}: Train={len(X_train)}, Test={len(X_test)}, Missing={missing_attack}")
            
            # Additional validation
            if len(X_test) == 0:
                logger.error(f"❌ Client {client_id} has no test data!")
            
        except Exception as e:
            logger.error(f"❌ Client {client_id} failed: {str(e)}")
    
    logger.info("✅ Zero-day setup validation complete")

if __name__ == "__main__":
    # Test the zero-day setup
    validate_zero_day_setup("Bot_IoT.csv", num_clients=5, label_col="category")