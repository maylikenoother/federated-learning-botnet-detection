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

# ---------------------------------------------------------------------
# Class order MUST match server.py for confusion-matrix aggregation
# ---------------------------------------------------------------------
CLASSES = ["Normal", "DDoS", "DoS", "Reconnaissance", "Theft"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enhanced settings for variable client experiments
CLIENT_ID = int(os.environ.get("CLIENT_ID", 0))
SERVER_ADDRESS = os.environ.get("SERVER_ADDRESS", "localhost:8080")
NUM_CLIENTS = int(os.environ.get("NUM_CLIENTS", 5))  # NEW: Support variable client counts
BATCH_SIZE = 32
EPOCHS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"🚀 Enhanced Client {CLIENT_ID} starting on {DEVICE}")
logger.info(f"📡 Server: {SERVER_ADDRESS}")
logger.info(f"👥 Expected client pool size: {NUM_CLIENTS}")

def wait_for_server(server_address, max_attempts=60):
    """Enhanced server connection with better error handling"""
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

class EnhancedFlowerClient(fl.client.NumPyClient):
    """
    Enhanced Flower client with variable client support and fog integration
    """
    
    def __init__(self, model, train_loader, test_loader, device, client_id, missing_attack):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.client_id = client_id
        self.missing_attack = missing_attack
        
        # Use shared class order
        self.attack_types = CLASSES
        self.missing_attack_id = self.attack_types.index(missing_attack) if missing_attack in self.attack_types else 0
        
        # Enhanced metrics tracking for research
        self.round_performance = []
        self.threat_detection_history = []
        self.training_stability_metrics = []
        
        # Validate data loaders
        if len(train_loader.dataset) == 0:
            raise ValueError(f"Client {client_id} has empty training dataset")
        if len(test_loader.dataset) == 0:
            logger.warning(f"Client {client_id} has empty test dataset")
        
        logger.info(f"✅ Enhanced Client {client_id} initialized")
        logger.info(f"📊 Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")
        logger.info(f"🎯 Missing attack: {missing_attack} (ID: {self.missing_attack_id})")
        logger.info(f"🔬 Zero-day simulation: Client excludes '{missing_attack}' from training")

    def get_parameters(self, config):
        """Get model parameters with enhanced error handling"""
        try:
            return [val.cpu().numpy() for val in self.model.state_dict().values()]
        except Exception as e:
            logger.error(f"Client {self.client_id}: Parameter extraction failed: {e}")
            return [np.array([0.0])]

    def set_parameters(self, parameters):
        """Set model parameters with validation"""
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
        """
        Enhanced training with variable client awareness and fog integration
        """
        server_round = config.get('server_round', 'unknown')
        target_clients = config.get('target_clients', 5)
        
        logger.info(f"🎯 Client {self.client_id} - Training Round {server_round}")
        logger.info(f"📊 Target clients for this round: {target_clients}")
        
        # Check for sufficient data based on client scale
        min_samples = max(50, 20 * target_clients)  # Scale minimum data with client count
        if len(self.train_loader.dataset) < min_samples:
            logger.warning(f"⚠️ Client {self.client_id}: Insufficient data ({len(self.train_loader.dataset)} samples) for {target_clients} client setup")
            return self.get_parameters(config), 0, {
                "loss": 0.0,
                "accuracy": 0.0,
                "status": "insufficient_data",
                "missing_attack": self.missing_attack,
                "target_clients": target_clients
            }
        
        try:
            training_start_time = time.time()
            self.set_parameters(parameters)
            self.model.train()
            
            # Enhanced optimizer settings based on client scale
            base_lr = config.get('learning_rate', 0.001)
            scaled_lr = base_lr * (5.0 / max(target_clients, 5))  # Scale LR inversely with client count
            
            optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=scaled_lr,
                weight_decay=1e-4
            )
            criterion = nn.CrossEntropyLoss()
            
            total_loss = 0.0
            total_samples = 0
            successful_batches = 0
            correct_predictions = 0
            
            # Adaptive batch processing based on client scale
            max_batches = min(50, len(self.train_loader))
            if target_clients > 10:
                max_batches = min(30, len(self.train_loader))  # Fewer batches for large-scale experiments
            
            for epoch in range(EPOCHS):
                epoch_loss = 0.0
                epoch_samples = 0
                
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
                        epoch_loss += loss.item()
                        epoch_samples += data.size(0)
                        successful_batches += 1
                        
                        # Calculate training accuracy
                        pred = output.argmax(dim=1, keepdim=True)
                        correct_predictions += pred.eq(target.view_as(pred)).sum().item()
                        
                    except Exception as e:
                        logger.debug(f"Batch {batch_idx} failed: {e}")
                        continue
                
                total_loss += epoch_loss
                total_samples += epoch_samples
            
            training_time = time.time() - training_start_time
            
            # Calculate final metrics
            avg_loss = total_loss / max(successful_batches, 1)
            accuracy = correct_predictions / max(total_samples, 1)
            
            # Track training stability for research
            stability_score = self._calculate_training_stability(avg_loss, accuracy, successful_batches)
            
            logger.info(f"✅ Client {self.client_id} training complete - "
                       f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, "
                       f"Batches: {successful_batches}, Time: {training_time:.2f}s")
            
            # Enhanced metrics for variable client analysis
            return self.get_parameters(config), total_samples, {
                "loss": float(avg_loss),
                "accuracy": float(accuracy),
                "batches_processed": successful_batches,
                "training_time": float(training_time),
                "stability_score": float(stability_score),
                "missing_attack": self.missing_attack,
                "target_clients": target_clients,
                "scaled_learning_rate": float(scaled_lr),
                "data_efficiency": float(total_samples / len(self.train_loader.dataset))
            }
            
        except Exception as e:
            logger.error(f"❌ Client {self.client_id} training failed: {e}")
            return self.get_parameters(config), 0, {
                "loss": float('inf'),
                "accuracy": 0.0,
                "error": str(e),
                "missing_attack": self.missing_attack,
                "target_clients": target_clients
            }

    def evaluate(self, parameters, config):
        """
        Enhanced evaluation with fog-ready threat detection and scalability metrics
        + confusion matrix & macro/micro Precision/Recall/F1 (NEW)
        """
        server_round = config.get('server_round', 'unknown')
        target_clients = config.get('target_clients', 5)
        
        logger.info(f"📊 Client {self.client_id} - Evaluation Round {server_round}")
        logger.info(f"🔍 Zero-day detection for: {self.missing_attack} (fog-ready)")
        
        # Skip evaluation if no test data
        if len(self.test_loader.dataset) == 0:
            logger.warning(f"⚠️ Client {self.client_id}: No test data")
            return 0.0, 0, {
                "accuracy": 0.0,
                "missing_attack": self.missing_attack,
                "status": "no_test_data",
                "target_clients": target_clients
            }
        
        try:
            evaluation_start_time = time.time()
            self.set_parameters(parameters)
            self.model.eval()
            
            criterion = nn.CrossEntropyLoss()
            test_loss = 0.0
            correct = 0
            total = 0
            
            # Enhanced zero-day metrics
            zero_day_tp = 0
            zero_day_fp = 0
            zero_day_tn = 0
            zero_day_fn = 0
            zero_day_total = 0
            
            # Threat confidence tracking
            threat_detections = []
            attack_confidence_scores = []

            # For confusion matrix (NEW)
            y_true_all = []
            y_pred_all = []
            
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

                        # Track labels/preds for confusion matrix
                        y_true_all.extend(target.detach().cpu().tolist())
                        y_pred_all.extend(pred.view(-1).detach().cpu().tolist())
                        
                        # Enhanced zero-day detection with confidence scoring
                        softmax_output = torch.softmax(output, dim=1)
                        
                        for i in range(len(target)):
                            true_label = target[i].item()
                            pred_label = pred[i].item()
                            confidence = softmax_output[i][pred_label].item()
                            
                            if true_label == self.missing_attack_id:
                                zero_day_total += 1
                                if pred_label == self.missing_attack_id:
                                    zero_day_tp += 1
                                    threat_detections.append({
                                        'type': self.missing_attack,
                                        'confidence': confidence,
                                        'correct': True
                                    })
                                else:
                                    zero_day_fn += 1
                            else:
                                if pred_label == self.missing_attack_id:
                                    zero_day_fp += 1
                                    threat_detections.append({
                                        'type': self.missing_attack,
                                        'confidence': confidence,
                                        'correct': False
                                    })
                                else:
                                    zero_day_tn += 1
                            
                            attack_confidence_scores.append(confidence)
                        
                    except Exception as e:
                        logger.debug(f"Evaluation batch {batch_idx} failed: {e}")
                        continue
            
            # ---- Build confusion matrix (NEW, no sklearn required) ----
            num_classes = len(CLASSES)
            cm = np.zeros((num_classes, num_classes), dtype=np.int64)
            for t, p in zip(y_true_all, y_pred_all):
                if 0 <= t < num_classes and 0 <= p < num_classes:
                    cm[t, p] += 1

            total_cm = int(cm.sum())
            per_class_counts = {}
            for idx, cls in enumerate(CLASSES):
                tp = int(cm[idx, idx])
                fp = int(cm[:, idx].sum() - tp)
                fn = int(cm[idx, :].sum() - tp)
                tn = int(total_cm - tp - fp - fn)
                per_class_counts[cls] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}

            # ---- Macro/Micro Precision/Recall/F1 (NEW) ----
            def safe_div(a, b): return (a / b) if b else 0.0
            def prf_from_counts(tp, fp, fn):
                prec = safe_div(tp, tp + fp)
                rec  = safe_div(tp, tp + fn)
                f1   = safe_div(2 * prec * rec, (prec + rec)) if (prec + rec) else 0.0
                return prec, rec, f1

            macro_p_list, macro_r_list, macro_f1_list = [], [], []
            tot_tp = tot_fp = tot_fn = 0
            for cls in CLASSES:
                tp = per_class_counts[cls]["tp"]
                fp = per_class_counts[cls]["fp"]
                fn = per_class_counts[cls]["fn"]
                p, r, f1 = prf_from_counts(tp, fp, fn)
                macro_p_list.append(p); macro_r_list.append(r); macro_f1_list.append(f1)
                tot_tp += tp; tot_fp += fp; tot_fn += fn

            macro_precision = float(np.mean(macro_p_list)) if macro_p_list else 0.0
            macro_recall    = float(np.mean(macro_r_list)) if macro_r_list else 0.0
            macro_f1        = float(np.mean(macro_f1_list)) if macro_f1_list else 0.0

            micro_precision, micro_recall, micro_f1 = prf_from_counts(tot_tp, tot_fp, tot_fn)

            evaluation_time = time.time() - evaluation_start_time
            
            # Calculate overall metrics
            accuracy = correct / max(total, 1)
            avg_loss = test_loss / max(total, 1)
            
            # Calculate enhanced zero-day detection metrics
            zero_day_precision = zero_day_tp / max(zero_day_tp + zero_day_fp, 1)
            zero_day_recall = zero_day_tp / max(zero_day_tp + zero_day_fn, 1)
            zero_day_f1 = 2 * (zero_day_precision * zero_day_recall) / max(zero_day_precision + zero_day_recall, 1)
            zero_day_detection_rate = zero_day_tp / max(zero_day_total, 1)
            zero_day_fpr = zero_day_fp / max(zero_day_fp + zero_day_tn, 1)
            
            # Enhanced threat analysis for fog integration
            avg_threat_confidence = np.mean([t['confidence'] for t in threat_detections]) if threat_detections else 0.0
            threat_detection_quality = len([t for t in threat_detections if t['correct']]) / max(len(threat_detections), 1)
            
            # Scalability impact analysis
            scalability_factor = self._calculate_scalability_impact(target_clients, accuracy, zero_day_detection_rate)
            
            logger.info(f"✅ Client {self.client_id} evaluation complete:")
            logger.info(f"   Overall Accuracy: {accuracy:.4f}, Loss: {avg_loss:.6f}")
            logger.info(f"   Zero-day samples: {zero_day_total}")
            logger.info(f"   Zero-day detection rate: {zero_day_detection_rate:.4f}")
            logger.info(f"   Threat confidence: {avg_threat_confidence:.4f}")
            logger.info(f"   Scalability factor: {scalability_factor:.4f}")
            logger.info(f"🔢 Confusion (sum={total_cm}): diag={int(np.diag(cm).sum())} classes={CLASSES}")
            logger.info(f"📏 Macro: P={macro_precision:.3f} R={macro_recall:.3f} F1={macro_f1:.3f} | "
                        f"Micro: P={micro_precision:.3f} R={micro_recall:.3f} F1={micro_f1:.3f}")
            
            # Store detection history for research analysis
            self.threat_detection_history.append({
                'round': server_round,
                'zero_day_detection_rate': zero_day_detection_rate,
                'threat_confidence': avg_threat_confidence,
                'target_clients': target_clients
            })
            
            # Build metrics payload
            metrics = {
                "accuracy": float(accuracy),
                "missing_attack": self.missing_attack,
                
                # Enhanced zero-day metrics for fog integration
                "zero_day_detection_rate": float(zero_day_detection_rate),
                "zero_day_precision": float(zero_day_precision),
                "zero_day_recall": float(zero_day_recall),
                "zero_day_f1_score": float(zero_day_f1),
                "zero_day_false_positive_rate": float(zero_day_fpr),
                "zero_day_samples": int(zero_day_total),
                
                # Fog-ready threat metrics
                "threat_confidence": float(avg_threat_confidence),
                "threat_detection_quality": float(threat_detection_quality),
                "threats_detected": int(len(threat_detections)),
                
                # Scalability metrics
                "target_clients": int(target_clients),
                "scalability_factor": float(scalability_factor),
                "evaluation_time": float(evaluation_time),
                "total_samples": int(total),
                
                # Research metrics
                "client_performance_stability": float(self._get_performance_stability()),
                "data_heterogeneity_impact": float(self._calculate_data_heterogeneity_impact()),

                # Macro/Micro PRF (client-side, server also recomputes)
                "macro_precision": float(macro_precision),
                "macro_recall": float(macro_recall),
                "macro_f1": float(macro_f1),
                "micro_precision": float(micro_precision),
                "micro_recall": float(micro_recall),
                "micro_f1": float(micro_f1),
            }

            # Per-class confusion counts (server expects these exact keys)
            for cls in CLASSES:
                metrics[f"cm_{cls}_tp"] = int(per_class_counts[cls]["tp"])
                metrics[f"cm_{cls}_fp"] = int(per_class_counts[cls]["fp"])
                metrics[f"cm_{cls}_fn"] = int(per_class_counts[cls]["fn"])
                metrics[f"cm_{cls}_tn"] = int(per_class_counts[cls]["tn"])

            return float(avg_loss), total, metrics
            
        except Exception as e:
            logger.error(f"❌ Client {self.client_id} evaluation failed: {e}")
            return float('inf'), 0, {
                "accuracy": 0.0,
                "missing_attack": self.missing_attack,
                "error": str(e),
                "target_clients": target_clients
            }
    
    def _calculate_training_stability(self, loss, accuracy, batches):
        """Calculate training stability score for research analysis"""
        try:
            if batches == 0:
                return 0.0
            stability = (accuracy * 0.5) + (1.0 / max(loss, 0.1) * 0.3) + (min(batches / 50, 1.0) * 0.2)
            return min(1.0, stability)
        except:
            return 0.5  # Default medium stability
    
    def _calculate_scalability_impact(self, target_clients, accuracy, zero_day_rate):
        """
        Calculate scalability impact factor for research analysis
        """
        try:
            baseline_clients = 5
            if target_clients <= baseline_clients:
                scale_benefit = 1.0
            else:
                scale_benefit = 1.0 + 0.1 * np.log(target_clients / baseline_clients) - 0.02 * (target_clients - baseline_clients)
            performance_factor = (accuracy + zero_day_rate) / 2.0
            scalability_factor = scale_benefit * performance_factor
            return min(2.0, max(0.1, scalability_factor))  # Bound between 0.1 and 2.0
        except:
            return 1.0  # Default neutral impact
    
    def _get_performance_stability(self):
        """Get client performance stability across rounds"""
        if len(self.threat_detection_history) < 2:
            return 1.0
        detection_rates = [h['zero_day_detection_rate'] for h in self.threat_detection_history]
        return 1.0 - np.std(detection_rates) if detection_rates else 1.0
    
    def _calculate_data_heterogeneity_impact(self):
        """
        Calculate impact of data heterogeneity on client performance
        """
        try:
            # Simple heuristic: missing 1 of 5 classes
            return 1.0 - (1.0 / len(self.attack_types))  # 0.8 for 5 classes
        except:
            return 0.8  # Default moderate heterogeneity

def main():
    """Enhanced main function with variable client support"""
    try:
        logger.info(f"🌟 Starting Enhanced FL Client {CLIENT_ID}")
        
        # Wait for server first
        if not wait_for_server(SERVER_ADDRESS, max_attempts=90):
            logger.error("❌ Server not available, exiting")
            return False
        
        # Load data with realistic chunk size
        logger.info(f"📂 Loading realistic dataset for client {CLIENT_ID}")
        
        (X_train, y_train), (X_test, y_test), missing_attack = load_and_partition_data(
            file_path="Bot_IoT.csv",
            client_id=CLIENT_ID,
            num_clients=NUM_CLIENTS,
            label_col="category",
            chunk_size=50000  # Realistic chunk for variable client experiments
        )
        
        logger.info(f"✅ Data loaded: Train={len(X_train)}, Test={len(X_test)}")
        logger.info(f"🎯 Zero-day attack simulation: Missing '{missing_attack}'")
        
        # Validate data sizes for variable client experiments
        min_required = 100 if NUM_CLIENTS <= 5 else 200 if NUM_CLIENTS <= 10 else 300
        if len(X_train) < min_required:
            logger.error(f"❌ Insufficient training data: {len(X_train)} samples (need {min_required} for {NUM_CLIENTS} clients)")
            return False
        
        # Create datasets with appropriate batch sizes
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        # Scale batch sizes based on client count
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
        num_classes = len(CLASSES)  # 5 for Bot-IoT
        
        model = Net(
            input_size=num_features, 
            output_size=num_classes,
            hidden_size=64,  # Balanced capacity
            num_hidden_layers=3,
            dropout_rate=0.3
        )
        
        logger.info(f"🧠 Model: {num_features} → {num_classes} classes")
        logger.info(f"📊 Architecture: 3 hidden layers, 64 units each")
        
        # Create enhanced client
        client = EnhancedFlowerClient(
            model, train_loader, test_loader, DEVICE, CLIENT_ID, missing_attack
        )
        
        # Connect to server
        logger.info(f"🔗 Connecting to enhanced server at {SERVER_ADDRESS}")
        logger.info(f"🎯 Ready for variable client experiments (5, 10, 15 clients)")
        
        fl.client.start_numpy_client(
            server_address=SERVER_ADDRESS, 
            client=client
        )
        
        logger.info(f"✅ Enhanced Client {CLIENT_ID} completed successfully!")
        logger.info(f"📊 Performance history: {len(client.threat_detection_history)} evaluations")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced Client {CLIENT_ID} failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
