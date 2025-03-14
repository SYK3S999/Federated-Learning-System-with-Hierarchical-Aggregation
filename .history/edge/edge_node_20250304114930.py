import logging
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import flwr as fl
from flwr.common import Parameters, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy import FedAvg
import numpy as np
from typing import Dict, Tuple, List
from flwr.common.typing import NDArrays, Scalar

class EdgeFormatter(logging.Formatter):
    def format(self, record):
        record.edge_id = getattr(record, 'edge_id', 'unknown')
        return super().format(record)

logger = logging.getLogger('edge_node')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = EdgeFormatter(
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
    logger.info(f"{stage} - CPU: {cpu_usage}%", extra={"edge_id": edge_id})

def evaluate_metrics_aggregation_fn(eval_metrics):
    accuracies = [metrics["accuracy"] * metrics["num_examples"] for _, metrics in eval_metrics]
    examples = [metrics["num_examples"] for _, metrics in eval_metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

def fit_config(server_round: int) -> Dict[str, Scalar]:
    return {"round": server_round, "local_epochs": 1}  # Add local_epochs for clients

class EdgeClient(fl.client.NumPyClient):
    def __init__(self, edge_id):
        self.edge_id = edge_id
        self.model = Net()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.parameters = None

    def set_parameters(self, parameters):
        logger.debug(f"Setting parameters, type: {type(parameters)}", extra={"edge_id": self.edge_id})
        if isinstance(parameters, Parameters):
            state_dict = parameters_to_ndarrays(parameters)
        elif isinstance(parameters, list):
            state_dict = parameters  # Assume it's already a list of NumPy arrays
        else:
            raise ValueError(f"Unsupported parameters type: {type(parameters)}")
        params_dict = zip(self.model.state_dict().keys(), state_dict)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
        self.parameters = parameters

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        logger.debug("Getting parameters", extra={"edge_id": self.edge_id})
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

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
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Starting Flower server on 0.0.0.0:8080 (attempt {attempt + 1}/{max_retries})", extra={"edge_id": self.edge_id})
                fl.server.start_server(
                    server_address="0.0.0.0:8080",
                    config=fl.server.ServerConfig(num_rounds=1),
                    strategy=client_strategy,
                )
                logger.info("Flower server started successfully", extra={"edge_id": self.edge_id})
                break
            except Exception as e:
                logger.warning(f"Server start failed: {str(e)}", extra={"edge_id": self.edge_id})
                if attempt == max_retries - 1:
                    raise
                time.sleep(5)

        elapsed = max(0, time.perf_counter() - start_time)
        log_resources(self.edge_id, f"Post-Aggregation (took {elapsed:.2f}s)")
        # Aggregate num_examples from clients (hardcoded for now, adjust if dynamic)
        num_examples = 15000  # 2 clients * 7500 (MNIST subset)
        return self.get_parameters({}), num_examples, {"aggregation_time": elapsed}

def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
    self.set_parameters(parameters)
    log_resources(self.edge_id, "Pre-Evaluation")
    # Placeholder: edges don’t train, just pass to clients
    return 0.0, 15000, {"accuracy": 0.0}

if __name__ == "__main__":
    edge_id = int(os.getenv("EDGE_ID", 0))
    try:
        edge = EdgeClient(edge_id)
        logger.info(f"Connecting to Cloud at cloud:8080", extra={"edge_id": edge_id})
        time.sleep(20)  # Keep delay for cloud
        fl.client.start_numpy_client(
            server_address="cloud:8080",
            client=edge,
            grpc_max_message_length=512 * 1024 * 1024  # 512MB, prevent message size issues
        )
    except Exception as e:
        logger.error(f"Failed: {str(e)}", extra={"edge_id": edge_id})
        raise