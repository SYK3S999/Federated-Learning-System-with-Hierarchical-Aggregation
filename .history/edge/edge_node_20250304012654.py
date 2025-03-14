import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters
import torch
import psutil
import time
import logging
from model import Net

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] Edge %(edge_id)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def log_resources(edge_id, phase):
    global net_io_start
    if "Pre" in phase:
        net_io_start = psutil.net_io_counters()
    cpu = psutil.cpu_percent(interval=0.1)
    net_io_end = psutil.net_io_counters()
    tx_mb = (net_io_end.bytes_sent - net_io_start.bytes_sent) / 1e6
    energy = cpu * 0.1 * 150 / 100
    logger.info(f"{phase} - CPU: {cpu:.1f}% | TX: {tx_mb:.2f} MB | Energy: {energy:.2f} J/s", extra={"edge_id": edge_id})

def evaluate_metrics_aggregation_fn(eval_metrics):
    accuracies = [m["accuracy"] for _, m in eval_metrics]
    return {"accuracy": sum(accuracies) / len(accuracies) if accuracies else 0}

def fit_config(server_round: int):
    return {"lr": 0.01, "momentum": 0.9, "local_epochs": 1}

class EdgeClient(fl.client.NumPyClient):
    def __init__(self, edge_id):
        self.edge_id = edge_id
        self.model = Net(10)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.parameters = None

    def set_parameters(self, parameters):
        self.parameters = parameters
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.Tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        log_resources(self.edge_id, "Pre-Aggregation")
        start_time = time.perf_counter()
        # Start Flower server for clients
        client_strategy = FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
            on_fit_config_fn=fit_config,
            initial_parameters=ndarrays_to_parameters(parameters),
        )
        fl.server.start_server(
            server_address=f"0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=1),
            strategy=client_strategy,
        )
        elapsed = max(0, time.perf_counter() - start_time)
        log_resources(self.edge_id, f"Post-Aggregation (took {elapsed:.2f}s)")
        return self.get_parameters({}), 2000, {}  # 2 clients * 1000 samples

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        log_resources(self.edge_id, "Pre-Evaluation")
        start_time = time.perf_counter()
        elapsed = max(0, time.perf_counter() - start_time)
        log_resources(self.edge_id, f"Post-Evaluation (took {elapsed:.2f}s)")
        return 0.0, 2000, {"accuracy": 0.98}  # Dummy for now

if __name__ == "__main__":
    import os
    edge_id = int(os.getenv("EDGE_ID", 0))
    try:
        edge = EdgeClient(edge_id)
        logger.info(f"Connecting to Cloud at cloud:8080", extra={"edge_id": edge_id})
        time.sleep(10)  # Wait for cloud
        fl.client.start_numpy_client(server_address="cloud:8080", client=edge)
    except Exception as e:
        logger.error(f"Failed: {str(e)}", extra={"edge_id": edge_id})
        raise