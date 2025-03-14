import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters
from typing import Dict
import torch
import psutil
import time
import logging
from dataset import prepare_dataset
from model import Net, test

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] Cloud - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class ResourceMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.net_io_start = None

    def get_stats(self, phase):
        if "Pre" in phase:
            self.net_io_start = psutil.net_io_counters()
        cpu = self.process.cpu_percent(interval=0.1)
        net_io_end = psutil.net_io_counters()
        tx_mb = (net_io_end.bytes_sent - self.net_io_start.bytes_sent) / 1e6
        energy = cpu * 0.1 * 150 / 100  # 150W GPU assumption
        logger.info(f"{phase} - CPU: {cpu:.1f}% | TX: {tx_mb:.2f} MB | Energy: {energy:.2f} J/s")
        return {"cpu": cpu, "tx_mb": tx_mb, "energy": energy}

def fit_config(server_round: int):
    return {"lr": 0.01, "momentum": 0.9, "local_epochs": 1, "server_round": server_round}

def evaluate_metrics_aggregation_fn(eval_metrics):
    accuracies = [m["accuracy"] * n for n, m in eval_metrics]
    total_examples = sum(n for n, _ in eval_metrics)
    return {"accuracy": sum(accuracies) / total_examples if total_examples else 0}

def fit_metrics_aggregation_fn(fit_metrics):
    total_examples = sum([n for n, _ in fit_metrics])
    avg_loss = sum([m["loss"] * n for n, m in fit_metrics]) / total_examples
    avg_accuracy = sum([m["accuracy"] * n for n, m in fit_metrics]) / total_examples
    return {"loss": avg_loss, "accuracy": avg_accuracy}

if __name__ == "__main__":
    try:
        _, _, testloader = prepare_dataset(num_partitions=4, batch_size=16)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = Net(num_classes=10)  # Already correct, just confirming
        net.to(device)
        monitor = ResourceMonitor()

        def evaluate_fn(server_round, parameters, config):
            pre_stats = monitor.get_stats(f"Round {server_round} Pre-Eval")
            start_time = time.perf_counter()
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            net.load_state_dict(state_dict, strict=True)
            loss, accuracy = test(net, testloader, device)
            elapsed = time.perf_counter() - start_time
            post_stats = monitor.get_stats(f"Round {server_round} Post-Eval (took {elapsed:.2f}s)")
            return loss, {"accuracy": accuracy, **pre_stats, **post_stats, "eval_time": elapsed}

        initial_params = [val.cpu().numpy() for _, val in net.state_dict().items()]
        strategy = FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
            evaluate_fn=evaluate_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            on_fit_config_fn=fit_config,
            initial_parameters=ndarrays_to_parameters(initial_params),
        )

        logger.info("Starting Cloud Server on 0.0.0.0:8080")
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=30),
            strategy=strategy,
            grpc_max_message_length=512 * 1024 * 1024,
        )
    except Exception as e:
        logger.error(f"Cloud Server failed: {str(e)}")
        raise