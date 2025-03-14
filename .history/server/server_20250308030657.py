import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters
from model import Net, test
from dataset import prepare_dataset
import torch
import psutil
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_metrics_aggregation_fn(eval_metrics):
    accuracies = [m["accuracy"] for _, m in eval_metrics]
    return {"accuracy": sum(accuracies) / len(accuracies)}

def fit_metrics_aggregation_fn(fit_metrics):
    total_examples = sum([n for n, _ in fit_metrics])
    avg_loss = sum([m["loss"] * n for n, m in fit_metrics]) / total_examples
    avg_accuracy = sum([m["accuracy"] * n for n, m in fit_metrics]) / total_examples
    return {"loss": avg_loss, "accuracy": avg_accuracy}

def fit_config(server_round: int):
    return {"lr": 0.01, "momentum": 0.9, "local_epochs": 1}

def log_resources(prefix):
    cpu_percent = psutil.cpu_percent()
    cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else "N/A"
    net_io = psutil.net_io_counters()
    tx_power = net_io.bytes_sent / 1e6  # Proxy for transmission power (MB)
    # Energy estimation (simplified: CPU % * time * avg wattage, e.g., 100W max)
    energy = cpu_percent * 0.1 * 100 / 100  # Joules/sec (tweak based on hardware)
    logger.info(f"{prefix} - CPU: {cpu_percent}% | Freq: {cpu_freq} MHz | TX Power: {tx_power:.2f} MB | Energy: {energy:.2f} J/s")

if __name__ == "__main__":
    _, _, testloader = prepare_dataset(num_partitions=2, batch_size=16)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net(10).to(device)
    
    def evaluate_fn(server_round, parameters, config):
        log_resources(f"Server Round {server_round} Pre-Eval")
        start_time = time.perf_counter()
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        net.load_state_dict(state_dict, strict=True)
        loss, accuracy = test(net, testloader, device)
        elapsed = max(0, time.perf_counter() - start_time)
        log_resources(f"Server Round {server_round} Post-Eval (took {elapsed:.2f}s)")
        return loss, {"accuracy": accuracy}

    initial_params = [val.cpu().numpy() for _, val in net.state_dict().items()]
    initial_parameters = ndarrays_to_parameters(initial_params)

    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=evaluate_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        on_fit_config_fn=fit_config,
        initial_parameters=initial_parameters,
    )

    logger.info("Starting FL Server...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=30),
        strategy=strategy,
    )