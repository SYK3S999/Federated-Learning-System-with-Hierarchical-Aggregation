import os
import time
import logging
import flwr as fl
from flwr.common import NDArrays, Scalar, ndarrays_to_parameters
from flwr.server.strategy import FedAvg
from typing import Dict, Tuple  # Added import
import torch
import psutil
from model import Net

logger = logging.getLogger('edge')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] Edge %(edge_id)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)
logger.handlers = [handler]

class ResourceMonitor:
    def __init__(self, edge_id):
        self.edge_id = edge_id
        self.process = psutil.Process()
        self.net_io_start = None

    def get_stats(self, stage):
        if "Pre" in stage:
            self.net_io_start = psutil.net_io_counters()
        cpu = self.process.cpu_percent(interval=0.1)
        net_io_end = psutil.net_io_counters()
        tx_mb = (net_io_end.bytes_sent - self.net_io_start.bytes_sent) / 1e6
        energy = cpu * 0.1 * 100 / 100  # 100W assumption
        logger.info(f"{stage} - CPU: {cpu:.1f}% | TX: {tx_mb:.2f} MB | Energy: {energy:.2f} J/s",
                   extra={"edge_id": self.edge_id})
        return {"cpu": cpu, "tx_mb": tx_mb, "energy": energy}

class EdgeClient(fl.client.NumPyClient):
    def __init__(self, edge_id):
        self.edge_id = edge_id
        self.model = Net()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.monitor = ResourceMonitor(edge_id)
    def fit_metrics_aggregation_fn(fit_metrics):
        total_examples = sum([n for n, _ in fit_metrics])
        avg_loss = sum([m["loss"] * n for n, m in fit_metrics]) / total_examples
        avg_accuracy = sum([m["accuracy"] * n for n, m in fit_metrics]) / total_examples
        return {"loss": avg_loss, "accuracy": avg_accuracy}

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: NDArrays):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        pre_stats = self.monitor.get_stats("Pre-Aggregation")
        start_time = time.perf_counter()

        strategy = FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
            on_fit_config_fn=lambda r: config,
            evaluate_metrics_aggregation_fn=lambda evals: {
                "accuracy": sum(m["accuracy"] * n for n, m in evals) / sum(n for n, _ in evals)
            },
            initial_parameters=ndarrays_to_parameters(parameters),
        )
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=10),
            strategy=strategy,
            grpc_max_message_length=512 * 1024 * 1024,
        )

        elapsed = time.perf_counter() - start_time
        post_stats = self.monitor.get_stats(f"Post-Aggregation (took {elapsed:.2f}s)")
        num_examples = 15000
        metrics = {**pre_stats, **post_stats, "aggregation_time": elapsed}
        return self.get_parameters({}), num_examples, metrics

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        pre_stats = self.monitor.get_stats("Pre-Evaluation")
        return 0.0, 15000, {"accuracy": 0.0, **pre_stats}

if __name__ == "__main__":
    edge_id = int(os.getenv("EDGE_ID", 0))
    edge = EdgeClient(edge_id)
    time.sleep(20)
    fl.client.start_numpy_client(
        server_address="cloud:8080",
        client=edge,
        grpc_max_message_length=512 * 1024 * 1024,
    )