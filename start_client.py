import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl

from model import Net, train, test
from partition_data import load_and_partition_data

CLIENT_ID = int(os.environ.get("CLIENT_ID", 0))
NUM_CLIENTS = 5
BATCH_SIZE = 64
EPOCHS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 Starting client {CLIENT_ID}")

X, y = load_and_partition_data(
    file_path="Bot_IoT.csv",
    client_id=CLIENT_ID,
    num_clients=NUM_CLIENTS,
    label_col="category",
    chunk_size=733700
)

dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = Net(input_size=X.shape[1], output_size=len(torch.unique(y))).to(DEVICE)

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config=None):
        return [val.cpu().numpy() for val in model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(model.state_dict().keys(), [torch.tensor(p) for p in parameters]))
        model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        print(f"🎯 Training client {CLIENT_ID}")
        self.set_parameters(parameters)
        train(model, train_loader, device=DEVICE, epochs=EPOCHS)
        return self.get_parameters(), len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(model, train_loader, device=DEVICE)
        return float(loss), len(train_loader.dataset), {"accuracy": float(accuracy)}

fl.client.start_numpy_client(server_address="localhost:8080", client=FlowerClient())
