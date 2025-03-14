import os
import time
import logging
import flwr as fl
from flwr.common import NDArrays, Scalar, ndarrays_to_parameters, Parameters
from flwr.server.strategy import FedAvg
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger('edge')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] Edge %(edge_id)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)
logger.handlers = [handler]

class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def log_resources(edge_id, stage):
    cpu_usage = os.popen("top -bn1 | grep 'Cpu(s)' | awk '{print $2}'").read().strip()
    tx_power = 0.0
    energy = float(cpu_usage) * 0.15 if cpu_usage else 0.0
    logger.info(f"{stage} - CPU: {cpu_usage}% | TX: {tx_power:.2f} MB | Energy: {energy:.2f} J/s", extra={"edge_id": edge_id})

def fit_config(server_round: int) -> Dict[str, Scalar]:
    return {"local_epochs": 1, "server_round": server_round}

def evaluate_metrics_aggregation_fn(eval_metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    total_examples = sum([num_examples for num_examples, _ in eval_metrics])
    weighted_acc = sum([m["accuracy"] * num_examples for num_examples, m in eval_metrics]) / total_examples
    return {"accuracy": weighted_acc}

class EdgeClient(fl.client.NumPyClient):
    def __init__(self, edge_id):
        self.edge_id = edge_id
        self.model = Net()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: NDArrays):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        log_resources(self.edge_id, "Pre-Aggregation")
        start_time = time.perf_counter()

        client_strategy = FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
            on_fit_config_fn=fit_config,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            initial_parameters=ndarrays_to_parameters(parameters),
        )
        logger.info("Starting Flower server on 0.0.0.0:8080", extra={"edge_id": self.edge_id})
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=1),
            strategy=client_strategy,
            grpc_max_message_length=512 * 1024 * 1024,
        )
        logger.info("Flower server completed round", extra={"edge_id": self.edge_id})

        elapsed = max(0, time.perf_counter() - start_time)
        log_resources(self.edge_id, f"Post-Aggregation (took {elapsed:.2f}s)")
        num_examples = 15000  # Adjust if dynamic
        return self.get_parameters({}), num_examples, {"aggregation_time": elapsed}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        log_resources(self.edge_id, "Pre-Evaluation")
        return 0.0, 15000, {"accuracy": 0.0}

if __name__ == "__main__":
    edge_id = int(os.getenv("EDGE_ID", 0))
    try:
        edge = EdgeClient(edge_id)
        logger.info("Edge initialized", extra={"edge_id": edge_id})
        logger.info("Connecting to Cloud at cloud:8080", extra={"edge_id": edge_id})
        time.sleep(20)
        fl.client.start_numpy_client(
            server_address="cloud:8080",
            client=edge,
            grpc_max_message_length=512 * 1024 * 1024,
        )
    except Exception as e:
        logger.error(f"Failed: {str(e)}", extra={"edge_id": edge_id})
        raise