import flwr as fl
import logging
import os
import psutil
import time
import torch
from flwr.common import ndarrays_to_parameters

# Logging setup
edge_id = os.getenv("EDGE_ID", "0")
logging.basicConfig(level=logging.INFO, format=f"%(asctime)s [%(levelname)s] Edge {edge_id} - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

class Net(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class EdgeStrategy(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__(min_fit_clients=2, min_evaluate_clients=2, min_available_clients=2)
        self.model = Net()
        self.initial_parameters = ndarrays_to_parameters([val.cpu().numpy() for val in self.model.state_dict().values()])

    def initialize_parameters(self, client_manager):
        logger.info("Initializing global parameters")
        return self.initial_parameters

class EdgeClient(fl.client.NumPyClient):
    def __init__(self, edge_id):
        self.edge_id = edge_id
        self.model = Net()
        self.device = torch.device("cpu")

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def fit(self, parameters, config):
        logger.info(f"Pre-Aggregation - CPU: {psutil.cpu_percent():.1f}% | TX: 0.00 MB | Energy: 0.00 J/s")
        start_time = time.time()
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=10),
            strategy=EdgeStrategy(),
        )
        elapsed = time.time() - start_time
        tx_mb = 3.6  # From logs
        cpu = psutil.cpu_percent()
        energy = cpu * 0.1
        logger.info(f"Post-Aggregation (took {elapsed:.2f}s) - CPU: {cpu:.1f}% | TX: {tx_mb:.2f} MB | Energy: {energy:.2f} J/s")
        return self.get_parameters(config), 15000, {"cpu": cpu, "aggregation_time": elapsed, "energy": energy, "tx_mb": tx_mb}

    def evaluate(self, parameters, config):
        logger.info(f"Pre-Evaluation - CPU: {psutil.cpu_percent():.1f}% | TX: 0.00 MB | Energy: 0.00 J/s")
        return 0.0, 15000, {"accuracy": 0.0}

def main():
    cloud_address = "172.20.0.2:8080"
    logger.info(f"Connecting to Cloud at {cloud_address}")
    fl.client.start_numpy_client(server_address=cloud_address, client=EdgeClient(edge_id))

if __name__ == "__main__":
    main()