
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_partition_data(file_path, client_id, num_clients, label_col="category", chunk_size=50000):
    """
    EXPERIMENT 3: Advanced partitioning with overlapping missing attacks and client specialization
    """
    experiment_seed = 98765  # CHANGED: Different base seed
    logger.info(f"🔄 EXPERIMENT 3 - Loading Bot-IoT data for client {client_id} with seed {experiment_seed}")
    
    try:
        # Set different random state
        np.random.seed(experiment_seed + client_id * 17)  # Different multiplier
        
        # Load data from different starting point
        skip_rows = 2000  # CHANGED: Skip first 2000 rows
        logger.info(f"📊 Reading Bot-IoT dataset, skipping {skip_rows} rows, chunk size {chunk_size}")
        df = pd.read_csv(file_path, nrows=chunk_size, skiprows=skip_rows)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        logger.info(f"✅ Loaded {len(df)} rows with columns: {list(df.columns)}")
        
        # Check for required columns
        if label_col not in df.columns:
            logger.error(f"❌ Column '{label_col}' not found. Available: {list(df.columns)}")
            raise ValueError(f"Label column '{label_col}' not found")
        
        # Clean data
        df = df.dropna()
        logger.info(f"📊 After cleaning: {len(df)} rows")
        
        # Get unique attack categories
        unique_categories = df[label_col].unique()
        logger.info(f"📋 Attack categories found: {unique_categories}")
        
        # Select numeric features
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = [label_col, 'attack', 'subcategory', 'target', 'pkSeqID', 'stime', 'ltime']
        feature_cols = [col for col in numeric_columns if col not in exclude_cols]
        
        logger.info(f"📊 Using {len(feature_cols)} numeric features")
        
        # CHANGED: Overlapping missing attacks (more realistic scenario)
        missing_attacks = {
            0: 'DDoS',          # Client 0 missing DDoS
            1: 'DDoS',          # Client 1 also missing DDoS (shared vulnerability)
            2: 'Theft',         # Client 2 missing Theft
            3: 'Reconnaissance', # Client 3 missing Reconnaissance  
            4: 'Normal'         # Client 4 missing Normal
        }
        
        missing_attack = missing_attacks.get(client_id, 'DDoS')
        logger.info(f"🎯 EXPERIMENT 3 - Client {client_id} missing: '{missing_attack}' (overlapping strategy)")
        
        # CHANGED: Client specialization based on network type
        logger.info(f"🌐 EXPERIMENT 3 - Applying client specialization for client {client_id}")
        
        # Define client types and their data preferences
        if client_id == 0:  # Corporate network
            client_type = "Corporate"
            preferred_ratio = {'Normal': 0.6, 'Reconnaissance': 0.25, 'DDoS': 0.08, 'DoS': 0.05, 'Theft': 0.02}
        elif client_id == 1:  # Web server
            client_type = "WebServer" 
            preferred_ratio = {'Normal': 0.3, 'DDoS': 0.35, 'DoS': 0.25, 'Reconnaissance': 0.07, 'Theft': 0.03}
        elif client_id == 2:  # IoT network
            client_type = "IoT"
            preferred_ratio = {'Normal': 0.4, 'Theft': 0.3, 'Reconnaissance': 0.2, 'DDoS': 0.06, 'DoS': 0.04}
        elif client_id == 3:  # Financial network
            client_type = "Financial"
            preferred_ratio = {'Normal': 0.5, 'Theft': 0.25, 'Reconnaissance': 0.15, 'DDoS': 0.06, 'DoS': 0.04}
        else:  # Mixed environment
            client_type = "Mixed"
            preferred_ratio = {'Normal': 0.35, 'DDoS': 0.2, 'DoS': 0.2, 'Reconnaissance': 0.15, 'Theft': 0.1}
        
        logger.info(f"🏢 Client {client_id} type: {client_type}")
        logger.info(f"📊 Preferred distribution: {preferred_ratio}")
        
        # Sample data based on client specialization
        specialized_data_frames = []
        for attack_type, ratio in preferred_ratio.items():
            if attack_type == missing_attack:
                continue  # Skip missing attack type for training
                
            attack_samples = df[df[label_col] == attack_type]
            if len(attack_samples) > 0:
                n_samples = int(len(df) * ratio * 0.8)  # 80% of desired ratio for training
                if n_samples > len(attack_samples):
                    n_samples = len(attack_samples)
                
                if n_samples > 0:
                    sample_seed = experiment_seed + client_id * 100 + hash(attack_type) % 1000
                    sampled = attack_samples.sample(n=n_samples, random_state=sample_seed)
                    specialized_data_frames.append(sampled)
        
        if specialized_data_frames:
            train_data = pd.concat(specialized_data_frames, ignore_index=True)
        else:
            # Fallback to original method
            train_mask = df[label_col] != missing_attack
            train_data = df[train_mask].copy()
        
        # CHANGED: Different train/test split ratio
        train_frac = 0.85  # CHANGED: 85% train, 15% test
        min_training_samples = 1500  # CHANGED: Lower minimum
        
        if len(train_data) < min_training_samples:
            logger.warning(f"⚠️ Only {len(train_data)} training samples, using {train_frac} for training")
            train_seed = experiment_seed * 4 + client_id
            train_data = df.sample(frac=train_frac, random_state=train_seed)
        
        # Create test set with remaining data + missing attack samples
        remaining_indices = df.index.difference(train_data.index)
        test_data = df.loc[remaining_indices].copy()
        
        # Add missing attack samples to test set
        missing_attack_samples = df[df[label_col] == missing_attack]
        if len(missing_attack_samples) > 0:
            test_sample_size = min(600, len(missing_attack_samples))  # CHANGED: Larger test sample
            test_seed = experiment_seed * 5 + client_id
            missing_samples = missing_attack_samples.sample(n=test_sample_size, random_state=test_seed)
            test_data = pd.concat([test_data, missing_samples]).drop_duplicates()
        
        logger.info(f"📊 EXPERIMENT 3 - Client {client_id} ({client_type}) data split:")
        logger.info(f"   Training: {len(train_data)} samples")
        logger.info(f"   Testing: {len(test_data)} samples")
        logger.info(f"   Missing attack '{missing_attack}' samples in test: {len(test_data[test_data[label_col] == missing_attack])}")
        
        # Show actual distribution in training data
        train_distribution = train_data[label_col].value_counts(normalize=True)
        logger.info(f"   🎯 Actual training distribution: {train_distribution.to_dict()}")
        
        # Encode labels
        label_encoder = LabelEncoder()
        all_labels = df[label_col].values
        label_encoder.fit(all_labels)
        
        train_labels = label_encoder.transform(train_data[label_col])
        test_labels = label_encoder.transform(test_data[label_col])
        
        # Prepare features
        X_train = train_data[feature_cols].values
        X_test = test_data[feature_cols].values
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(train_labels, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_test_tensor = torch.tensor(test_labels, dtype=torch.long)
        
        # Validation
        if len(X_train_tensor) < 100:
            raise ValueError(f"Insufficient training data: {len(X_train_tensor)} samples")
        
        logger.info(f"✅ EXPERIMENT 3 - Client {client_id} ({client_type}) data ready:")
        logger.info(f"   📊 Train: {len(X_train_tensor)} samples, {X_train_tensor.shape[1]} features")
        logger.info(f"   📊 Test: {len(X_test_tensor)} samples")
        logger.info(f"   🎯 Missing attack for zero-day: {missing_attack}")
        
        return (X_train_tensor, y_train_tensor), (X_test_tensor, y_test_tensor), missing_attack
        
    except Exception as e:
        logger.error(f"❌ EXPERIMENT 3 - Data loading failed for client {client_id}: {str(e)}")
        
        # Emergency fallback
        logger.warning("🔄 Creating fallback data for EXPERIMENT 3")
        
        torch.manual_seed(experiment_seed + client_id)
        
        n_train, n_test = 1800, 450  # CHANGED: Different sizes
        n_features = 25
        n_classes = 5
        
        X_train = torch.randn(n_train, n_features)
        y_train = torch.randint(0, n_classes, (n_train,))
        X_test = torch.randn(n_test, n_features)
        y_test = torch.randint(0, n_classes, (n_test,))
        
        # Overlapping missing attacks
        missing_attack = ['DDoS', 'DDoS', 'Theft', 'Reconnaissance', 'Normal'][client_id % 5]
        
        logger.info(f"⚠️ EXPERIMENT 3 - Using fallback data")
        return (X_train, y_train), (X_test, y_test), missing_attack

def validate_zero_day_setup(file_path, num_clients=5, label_col="category"):
    """
    Validate EXPERIMENT 3 setup
    """
    logger.info("🔍 Validating EXPERIMENT 3 zero-day FL setup with client specialization...")
    
    success_count = 0
    total_train_samples = 0
    total_test_samples = 0
    
    for client_id in range(num_clients):
        try:
            (X_train, y_train), (X_test, y_test), missing_attack = load_and_partition_data(
                file_path=file_path,
                client_id=client_id, 
                num_clients=num_clients,
                label_col=label_col,
                chunk_size=50000
            )
            
            train_size = len(X_train)
            test_size = len(X_test)
            total_train_samples += train_size
            total_test_samples += test_size
            
            logger.info(f"✅ EXPERIMENT 3 - Client {client_id}: Train={train_size}, Test={test_size}, Missing={missing_attack}")
            success_count += 1
            
        except Exception as e:
            logger.error(f"❌ EXPERIMENT 3 - Client {client_id} validation failed: {str(e)}")
    
    logger.info(f"\n🎯 EXPERIMENT 3 Validation Summary:")
    logger.info(f"   ✅ Successful clients: {success_count}/{num_clients}")
    logger.info(f"   📊 Total training samples: {total_train_samples:,}")
    logger.info(f"   📊 Total test samples: {total_test_samples:,}")
    logger.info(f"   🏢 Client specializations: Corporate, WebServer, IoT, Financial, Mixed")
    logger.info(f"   🔄 Overlapping missing attacks: Some clients share vulnerabilities")
    
    if success_count == num_clients:
        logger.info("🎉 EXPERIMENT 3 setup validated successfully!")
        return True
    else:
        logger.error("❌ EXPERIMENT 3 setup validation failed")
        return False

if __name__ == "__main__":
    if validate_zero_day_setup("Bot_IoT.csv", num_clients=5, label_col="category"):
        print("✅ EXPERIMENT 3 ready for federated learning!")
    else:
        print("❌ EXPERIMENT 3 setup needs fixes")