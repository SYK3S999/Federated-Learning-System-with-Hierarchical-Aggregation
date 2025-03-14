import os
import flwr as fl
from flwr.common import ndarrays_to_parameters
import torch
import psutil
import time
import logging
from model import Net

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EdgeClient(fl.client.NumPyClient):
    def __init__(self, edge_id, clients):
        self.edge_id = edge_id
        self.clients = clients  # List of client IDs under this edge
        self.model = Net(10)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.parameters = None

    def log_resources(self, phase):
        cpu = psutil.cpu_percent()
        tx = psutil.net_io_counters().bytes_sent / 1e6
        energy = cpu * 0.1 * 100 / 100
        logger.info(f"Edge {self.edge_id} {phase} - CPU: {cpu}% | TX: {tx:.2f} MB | Energy: {energy:.2f} J/s")

    def set_parameters(self, parameters):
        self.parameters = parameters
        # Forward to clients (simulated here; real HFL would aggregate later)
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.Tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.log_resources("Pre-Aggregation")
        start_time = time.perf_counter()
        # Simulate aggregation from clients (replace with real client comms later)
        time.sleep(2)  # Placeholder for client aggregation
        elapsed = max(0, time.perf_counter() - start_time)
        self.log_resources(f"Post-Aggregation (took {elapsed:.2f}s)")
        return self.get_parameters({}), len(self.clients) * 1000, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.log_resources("Pre-Evaluation")
        start_time = time.perf_counter()
        # Placeholder eval (real HFL would use client results)
        elapsed = max(0, time.perf_counter() - start_time)
        self.log_resources(f"Post-Evaluation (took {elapsed:.2f}s)")
        return 0.0, 1000, {"accuracy": 0.98}  # Dummy values

if __name__ == "__main__":
    edge_id = int(os.getenv("EDGE_ID", 0))
    clients = [0, 1] if edge_id == 0 else [2, 3]  # Edge 0: clients 0,1; Edge 1: clients 2,3
    time.sleep(10)  # Wait for cloud
    edge = EdgeClient(edge_id, clients)
    logger.info(f"Edge {edge_id} starting...")
    fl.client.start_numpy_client(server_address="cloud:8080", client=edge)