import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import os
import logging
import psutil
import time

# Custom formatter with safe cid handling
class ClientFormatter(logging.Formatter):
    def format(self, record):
        if not hasattr(record, 'cid'):
            record.cid = 'unknown'  # Fallback for Flower logs
        return super().format(record)

# Set up logger specific to this module
logger = logging.getLogger('client')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = ClientFormatter(
    fmt="%(asctime)s [%(levelname)s] Client %(cid)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)
logger.handlers = [handler]

class Net(nn.Module):
    def __init__(self, num_classes):
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

net_io_start = None

def log_resources(cid, phase):
    global net_io_start
    if "Pre" in phase:
        net_io_start = psutil.net_io_counters()
    cpu = psutil.cpu_percent(interval=0.1)
    net_io_end = psutil.net_io_counters()
    tx_mb = (net_io_end.bytes_sent - net_io_start.bytes_sent) / 1e6
    energy = cpu * 0.1 * 150 / 100  # 150W GPU assumption
    logger.info(f"{phase} - CPU: {cpu:.1f}% | TX: {tx_mb:.2f} MB | Energy: {energy:.2f} J/s", extra={"cid": cid})

def load_data(cid, num_clients=4):
    transform = ToTensor()
    trainset = MNIST(root="./data", train=True, download=True, transform=transform)
    testset = MNIST(root="./data", train=False, download=True, transform=transform)
    partition_size = len(trainset) // num_clients
    start_idx = cid * partition_size
    end_idx = start_idx + partition_size
    train_subset = torch.utils.data.Subset(trainset, range(start_idx, end_idx))
    return DataLoader(train_subset, batch_size=32), DataLoader(testset, batch_size=32)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, num_classes, cid):
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Net(num_classes).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        self.cid = cid

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        log_resources(self.cid, "Pre-Training")
        start_time = time.perf_counter()
        self.model.train()
        for _ in range(config["local_epochs"]):
            for data, target in self.trainloader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        elapsed = max(0, time.perf_counter() - start_time)
        log_resources(self.cid, f"Post-Training (took {elapsed:.2f}s)")
        return self.get_parameters({}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        log_resources(self.cid, "Pre-Evaluation")
        start_time = time.perf_counter()
        self.model.eval()
        loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for data, target in self.valloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        elapsed = max(0, time.perf_counter() - start_time)
        log_resources(self.cid, f"Post-Evaluation (took {elapsed:.2f}s)")
        return loss / len(self.valloader), total, {"accuracy": correct / total}

if __name__ == "__main__":
    cid = int(os.getenv("CID", 0))
    edge_id = cid // 2
    trainloader, valloader = load_data(cid)
    max_retries = 5
    retry_delay = 10  # seconds
    try:
        client = FlowerClient(trainloader, valloader, num_classes=10, cid=cid)
        logger.info(f"Connecting to Edge {edge_id} at edge{edge_id}:8080", extra={"cid": cid})
        for attempt in range(max_retries):
            try:
                time.sleep(20 if attempt == 0 else retry_delay)  # Initial 20s, then 10s retries
                fl.client.start_numpy_client(server_address=f"edge{edge_id}:8080", client=client)
                break  # Success, exit loop
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}", extra={"cid": cid})
                if attempt == max_retries - 1:
                    raise  # Last attempt failed, re-raise
    except Exception as e:
        logger.error(f"Failed after {max_retries} retries: {str(e)}", extra={"cid": cid})
        raise