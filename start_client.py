import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
import flwr as fl
import logging
from collections import OrderedDict
import traceback
import time
import numpy as np

from model import Net
from partition_data import load_and_partition_data

# ------------------ Configuration ------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CLIENT_ID = int(os.environ.get("CLIENT_ID", 0))
NUM_CLIENTS = 5
BATCH_SIZE = 16  # Reduced for edge-device memory limits
EPOCHS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GLOBAL_CLASSES = ['DDoS', 'DoS', 'Normal', 'Reconnaissance', 'Theft']
NUM_GLOBAL_CLASSES = len(GLOBAL_CLASSES)

logger.info(f"🚀 Starting zero-day simulation client {CLIENT_ID} on device: {DEVICE}")

# ------------------ Flower Client Class ------------------
class ZeroDayFlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device, client_id, missing_attack):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.client_id = client_id
        self.missing_attack = missing_attack
        self.missing_attack_idx = GLOBAL_CLASSES.index(missing_attack)
        
        logger.info(f"Client {client_id} initialized - Missing attack: {missing_attack} (index {self.missing_attack_idx})")
        logger.info(f"Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = OrderedDict(zip(self.model.state_dict().keys(), [torch.tensor(v) for v in parameters]))
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        server_round = config.get('server_round', 'unknown')
        logger.info(f"🎯 Client {self.client_id} - Training Round {server_round}")

        try:
            if len(self.train_loader.dataset) < 20:
                logger.warning(f"⚠️ Not enough data to train this round, skipping")
                return self.get_parameters(config), 0, {"skipped": True}

            self.set_parameters(parameters)
            self.model.train()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=config.get('learning_rate', 0.001))
            criterion = nn.CrossEntropyLoss()
            total_loss, total_samples = 0.0, 0

            for epoch in range(config.get('local_epochs', EPOCHS)):
                epoch_loss, epoch_samples = 0.0, 0
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    if batch_idx >= 10:  # ✅ Limit training batches per round
                        break
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = criterion(output, target)

                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning("Invalid loss, skipping batch")
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                    epoch_samples += data.size(0)

                total_loss += epoch_loss
                total_samples += epoch_samples

                if epoch_samples > 0:
                    logger.info(f"Client {self.client_id} - Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss / epoch_samples:.4f}")

            avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
            return self.get_parameters(config), len(self.train_loader.dataset), {"loss": float(avg_loss), "missing_attack": self.missing_attack}

        except Exception as e:
            logger.error(f"Client {self.client_id} training failed: {e}")
            logger.error(traceback.format_exc())
            raise e

    def evaluate(self, parameters, config):
        logger.info(f"📊 Client {self.client_id} - Evaluating (Zero-day: {self.missing_attack})")
        try:
            self.set_parameters(parameters)
            self.model.eval()
            criterion = nn.CrossEntropyLoss()

            total_loss, correct, total = 0.0, 0, 0
            zero_day_tp = zero_day_fp = zero_day_fn = zero_day_tn = zero_day_samples = 0

            with torch.no_grad():
                for data, target in self.test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    loss = criterion(output, target)

                    if torch.isnan(loss) or torch.isinf(loss):
                        continue

                    total_loss += loss.item() * data.size(0)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)

                    for i in range(len(target)):
                        true_label, pred_label = target[i].item(), pred[i].item()
                        if true_label == self.missing_attack_idx:
                            zero_day_samples += 1
                            if pred_label == self.missing_attack_idx:
                                zero_day_tp += 1
                            else:
                                zero_day_fn += 1
                        else:
                            if pred_label == self.missing_attack_idx:
                                zero_day_fp += 1
                            else:
                                zero_day_tn += 1

            accuracy = correct / total if total > 0 else 0.0
            avg_loss = total_loss / total if total > 0 else float('inf')

            zero_day_metrics = {
                "zero_day_detection_rate": zero_day_tp / zero_day_samples if zero_day_samples > 0 else 0.0,
                "zero_day_fp_rate": zero_day_fp / (zero_day_fp + zero_day_tn) if (zero_day_fp + zero_day_tn) > 0 else 0.0,
                "zero_day_accuracy": (zero_day_tp + zero_day_tn) / (zero_day_tp + zero_day_fp + zero_day_fn + zero_day_tn)
                if (zero_day_tp + zero_day_fp + zero_day_fn + zero_day_tn) > 0 else 0.0,
                "zero_day_samples": zero_day_samples,
                "missing_attack": self.missing_attack
            }

            logger.info(f"Client {self.client_id} Eval - Acc: {accuracy:.4f}, Loss: {avg_loss:.4f}, Zero-Day Samples: {zero_day_samples}")
            return float(avg_loss), len(self.test_loader.dataset), {"accuracy": accuracy, **zero_day_metrics}

        except Exception as e:
            logger.error(f"Client {self.client_id} evaluation failed: {e}")
            logger.error(traceback.format_exc())
            return float('inf'), 0, {"accuracy": 0.0}

# ------------------ Main ------------------
def main():
    try:
        logger.info(f"📦 Loading client {CLIENT_ID} data...")
        (X_train, y_train), (X_test, y_test), missing_attack = load_and_partition_data(
            file_path="Bot_IoT.csv",
            client_id=CLIENT_ID,
            num_clients=NUM_CLIENTS,
            label_col="category",
            chunk_size=146740
        )

        if len(X_train) < 100:
            logger.warning(f"⚠️ Very few training samples ({len(X_train)})")

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=min(BATCH_SIZE, len(train_dataset)), shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=min(BATCH_SIZE, len(test_dataset)), shuffle=False)

        logger.info(f"✔️ Data loaded - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
        logger.info(f"📌 Missing attack type: {missing_attack}")

        model = Net(input_size=X_train.shape[1], output_size=NUM_GLOBAL_CLASSES)
        client = ZeroDayFlowerClient(model, train_loader, test_loader, DEVICE, CLIENT_ID, missing_attack)

        # Auto-reconnect if server isn't ready yet
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Connecting to server (attempt {attempt+1}/{max_retries})")
                fl.client.start_numpy_client(server_address="localhost:8080", client=client)
                break
            except Exception as e:
                logger.error(f"Connection failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    raise e

    except Exception as e:
        logger.error(f"❌ Client {CLIENT_ID} startup failed: {e}")
        logger.error(traceback.format_exc())
        raise e

if __name__ == "__main__":
    main()
