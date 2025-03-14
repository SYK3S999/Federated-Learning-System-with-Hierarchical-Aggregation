import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters
import torch
import psutil
import time
import logging
from dataset import prepare_dataset
from model import Net

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_resources(prefix):
    cpu = psutil.cpu_percent()
    tx = psutil.net_io_counters().bytes_sent / 1e6
    energy = cpu * 0.1 * 100 / 100
    logger.info(f"{prefix} - CPU: {cpu}% | TX: {tx:.2f} MB | Energy: {energy:.2f} J/s")

def evaluate_metrics_aggregation_fn(eval_metrics):
    accuracies = [m["accuracy"] for _, m in eval_metrics]
    return {"accuracy": sum(accuracies) / len(accuracies)}

def fit_config(server_round: int):
    return {"lr": 0.01, "momentum": 0.9, "local_epochs": 1}

if __name__ == "__main__":
    _, _, testloader = prepare_dataset(num_partitions=2, batch_size=16)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net(10).to(device)

    def evaluate_fn(server_round, parameters, config):
        log_resources(f"Cloud Round {server_round} Pre-Eval")
        start_time = time.perf_counter()
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        net.load_state_dict(state_dict, strict=True)
        loss, accuracy = test(net, testloader, device)
        elapsed = max(0, time.perf_counter() - start_time)
        log_resources(f"Cloud Round {server_round} Post-Eval (took {elapsed:.2f}s)")
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

    logger.info("Starting Cloud Server...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=30),
        strategy=strategy,
    )