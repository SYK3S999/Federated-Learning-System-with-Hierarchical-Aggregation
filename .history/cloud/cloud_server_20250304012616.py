import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters
import torch
import psutil
import time
import logging
from dataset import prepare_dataset
from model import Net, test

# Consistent logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] Cloud - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def log_resources(phase):
    """Log resources with delta TX for this task."""
    global net_io_start
    if "Pre" in phase:
        net_io_start = psutil.net_io_counters()
    cpu = psutil.cpu_percent(interval=0.1)
    net_io_end = psutil.net_io_counters()
    tx_mb = (net_io_end.bytes_sent - net_io_start.bytes_sent) / 1e6
    energy = cpu * 0.1 * 150 / 100  # Assume 150W GPU, J/s
    logger.info(f"{phase} - CPU: {cpu:.1f}% | TX: {tx_mb:.2f} MB | Energy: {energy:.2f} J/s")

def evaluate_metrics_aggregation_fn(eval_metrics):
    accuracies = [m["accuracy"] for _, m in eval_metrics]
    return {"accuracy": sum(accuracies) / len(accuracies) if accuracies else 0}

def fit_config(server_round: int):
    return {"lr": 0.01, "momentum": 0.9, "local_epochs": 1}

if __name__ == "__main__":
    try:
        _, _, testloader = prepare_dataset(num_partitions=4, batch_size=16)  # Full test set
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = Net(10).to(device)

        def evaluate_fn(server_round, parameters, config):
            log_resources(f"Round {server_round} Pre-Eval")
            start_time = time.perf_counter()
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            net.load_state_dict(state_dict, strict=True)
            loss, accuracy = test(net, testloader, device)
            elapsed = max(0, time.perf_counter() - start_time)
            log_resources(f"Round {server_round} Post-Eval (took {elapsed:.2f}s)")
            return loss, {"accuracy": accuracy}

        initial_params = [val.cpu().numpy() for _, val in net.state_dict().items()]
        initial_parameters = ndarrays_to_parameters(initial_params)

        strategy = FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=2,  # 2 edges
            min_evaluate_clients=2,
            min_available_clients=2,
            evaluate_fn=evaluate_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            on_fit_config_fn=fit_config,
            initial_parameters=initial_parameters,
        )

        logger.info("Starting Cloud Server on 0.0.0.0:8080")
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=30),
            strategy=strategy,
        )
    except Exception as e:
        logger.error(f"Cloud Server failed: {str(e)}")
        raise