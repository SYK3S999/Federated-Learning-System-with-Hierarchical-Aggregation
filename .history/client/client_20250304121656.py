import os
import time
import logging
import flwr as fl
from flwr.common import NDArrays, Scalar
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import numpy as np

logger = logging.getLogger('client')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] Client %(client_id)s - %(message)s",
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

def log_resources(client_id, stage):
    cpu_usage = os.popen("top -bn1 | grep 'Cpu(s)' | awk '{print $2}'").read().strip()
    tx_power = 0.0
    energy = float(cpu_usage) * 0.15 if cpu_usage else 0.0
    logger.info(f"{stage} - CPU: {cpu_usage}% | TX: {tx_power:.2f} MB | Energy: {energy:.2f} J/s", extra={"client_id": client_id})

def load_data(client_id):
    transform = ToTensor()
    trainset = MNIST(root='./data', train=True, download=True, transform=transform)
    testset = MNIST(root='./data', train=False, download=True, transform=transform)
    partition_size = len(trainset) // 4
    start_idx = client_id * partition_size
    end_idx = start_idx + partition_size
    train_subset = torch.utils.data.Subset(trainset, range(start_idx, end_idx))
    return DataLoader(train_subset, batch_size=32), DataLoader(testset, batch_size=32)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.model = Net()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.trainloader, self.testloader = load_data(client_id)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: NDArrays):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        log_resources(self.client_id, "Pre-Training")
        start_time = time.perf_counter()

        self.model.train()
        local_epochs = config.get("local_epochs", 1)
        for _ in range(local_epochs):
            for data, target in self.trainloader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

        elapsed = time.perf_counter() - start_time
        num_examples = len(self.trainloader.dataset)
        log_resources(self.client_id, f"Post-Training (took {elapsed:.2f}s)")
        return self.get_parameters({}), num_examples, {"training_time": elapsed}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        log_resources(self.client_id, "Pre-Evaluation")
        start_time = time.perf_counter()

        self.model.eval()
        loss, correct = 0, 0
        num_examples = 0
        with torch.no_grad():
            for data, target in self.testloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += self.criterion(output, target).item() * len(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                num_examples += len(data)

        loss /= num_examples
        accuracy = correct / num_examples
        elapsed = time.perf_counter() - start_time
        log_resources(self.client_id, f"Post-Evaluation (took {elapsed:.2f}s)")
        return loss, num_examples, {"accuracy": accuracy, "evaluation_time": elapsed}

if __name__ == "__main__":
    client_id = int(os.getenv("CLIENT_ID", 0))
    edge_id = client_id % 2
    edge_addr = f"edge{edge_id}:8080"
    max_retries = 5
    initial_delay = 60
    logger.info(f"Initial delay of {initial_delay}s before connecting to Edge {edge_id}", extra={"client_id": client_id})
    time.sleep(initial_delay)
    for attempt in range(max_retries):
        try:
            logger.info(f"Connecting to Edge {edge_id} at {edge_addr}", extra={"client_id": client_id})
            fl.client.start_numpy_client(
                server_address=edge_addr,
                client=FlowerClient(client_id),
                grpc_max_message_length=512 * 1024 * 1024,
            )
            break
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}", extra={"client_id": client_id})
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                logger.error(f"All attempts failed", extra={"client_id": client_id})
                raise