import os
import time
import logging
import flwr as fl
from flwr.common import NDArrays, Scalar
from typing import Dict, Tuple
import torch
import torch.nn as nn
import psutil
from dataset import prepare_dataset
from model import Net, test

logger = logging.getLogger('client')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(fmt="%(asctime)s [%(levelname)s] Client %(client_id)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logger.handlers = [handler]

class ResourceMonitor:
    def __init__(self, client_id):
        self.client_id = client_id
        self.process = psutil.Process()
        self.net_io_start = None

    def get_stats(self, stage):
        if "Pre" in stage:
            self.net_io_start = psutil.net_io_counters()
        cpu = self.process.cpu_percent(interval=0.1)
        net_io_end = psutil.net_io_counters()
        tx_mb = (net_io_end.bytes_sent - self.net_io_start.bytes_sent) / 1e6
        energy = cpu * 0.1 * 50 / 100  # 50W assumption
        logger.info(f"{stage} - CPU: {cpu:.1f}% | TX: {tx_mb:.2f} MB | Energy: {energy:.2f} J/s", extra={"client_id": self.client_id})
        return {"cpu": cpu, "tx_mb": tx_mb, "energy": energy}

class FlowerClient(fl.client.NumPyClient):
    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        pre_stats = self.monitor.get_stats("Pre-Training")
        start_time = time.perf_counter()

        self.model.train()
        local_epochs = config.get("local_epochs", 1)
        total_loss = 0.0
        correct = 0
        total = 0
        for _ in range(local_epochs):
            for data, target in self.trainloader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        elapsed = time.perf_counter() - start_time
        post_stats = self.monitor.get_stats(f"Post-Training (took {elapsed:.2f}s)")
        num_examples = len(self.trainloader.dataset)
        avg_loss = total_loss / len(self.trainloader)
        accuracy = correct / total
        metrics = {**pre_stats, **post_stats, "training_time": elapsed, "loss": avg_loss, "accuracy": accuracy}
        return self.get_parameters({}), num_examples, metrics



    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: NDArrays):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        pre_stats = self.monitor.get_stats("Pre-Training")
        start_time = time.perf_counter()

        self.model.train()
        local_epochs = config.get("local_epochs", 1)
        total_loss = 0.0
        correct = 0
        total = 0
        for _ in range(local_epochs):
            for data, target in self.trainloader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        elapsed = time.perf_counter() - start_time
        post_stats = self.monitor.get_stats(f"Post-Training (took {elapsed:.2f}s)")
        num_examples = len(self.trainloader.dataset)
        avg_loss = total_loss / len(self.trainloader)
        accuracy = correct / total
        metrics = {**pre_stats, **post_stats, "training_time": elapsed, "loss": avg_loss, "accuracy": accuracy}
        return self.get_parameters({}), num_examples, metrics
    

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        pre_stats = self.monitor.get_stats("Pre-Evaluation")
        try:
            loss, accuracy = test(self.model, self.trainloader, self.device)
            logger.info(f"Evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}", extra={"client_id": self.client_id})
            return float(loss), len(self.trainloader.dataset), {"accuracy": accuracy, **pre_stats}
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}", extra={"client_id": self.client_id})
            return float("inf"), len(self.trainloader.dataset), {"accuracy": 0.0, **pre_stats}

if __name__ == "__main__":
    client_id = int(os.getenv("CLIENT_ID", 0))
    edge_id = client_id % 2
    edge_addr = f"edge{edge_id}:8080"
    logger.info(f"Connecting to {edge_addr}", extra={"client_id": client_id})
    fl.client.start_numpy_client(
        server_address=edge_addr,
        client=FlowerClient(client_id),
        grpc_max_message_length=512 * 1024 * 1024,
    )