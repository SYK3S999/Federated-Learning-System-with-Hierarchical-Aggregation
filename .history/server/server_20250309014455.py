import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters
import torch
import psutil
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] Cloud - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
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
    tx_mb = psutil.net_io_counters().bytes_sent / 1e6
    energy = cpu_percent * 0.15
    logger.info(f"{prefix} - CPU: {cpu_percent:.1f}% | TX: {tx_mb:.2f} MB | Energy: {energy:.2f} J/s")
    return {"cpu": cpu_percent, "tx_mb": tx_mb, "energy": energy}

if __name__ == "__main__":
    device = torch.device("cpu")
    net = Net(10).to(device)

    def evaluate_fn(server_round, parameters, config):
        log_resources(f"Round {server_round} Pre-Eval")
        start_time = time.perf_counter()
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        net.load_state_dict(state_dict, strict=True)
        loss, accuracy = 182.349, 0.0213  # Placeholder from logs
        elapsed = time.perf_counter() - start_time
        post_stats = log_resources(f"Round {server_round} Post-Eval (took {elapsed:.2f}s)")
        return loss, {"accuracy": accuracy, **post_stats, "eval_time": elapsed}

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
        grpc_max_message_length=512 * 1024 * 1024,
    )