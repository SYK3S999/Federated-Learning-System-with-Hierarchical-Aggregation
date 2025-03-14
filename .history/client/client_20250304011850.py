import flwr as fl
import torch
from collections import OrderedDict
import os
import time
import psutil
import logging
from model import Net, test, train
from dataset import prepare_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, num_classes, cid):
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = Net(num_classes)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cid = cid

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def log_resources(self, phase):
        cpu_percent = psutil.cpu_percent()
        cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else "N/A"
        net_io = psutil.net_io_counters()
        tx_power = net_io.bytes_sent / 1e6  # Proxy (MB)
        energy = cpu_percent * 0.1 * 100 / 100  # Joules/sec
        logger.info(f"Client {self.cid} {phase} - CPU: {cpu_percent}% | Freq: {cpu_freq} MHz | TX Power: {tx_power:.2f} MB | Energy: {energy:.2f} J/s")

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        lr = config["lr"]
        momentum = config["momentum"]
        epochs = config["local_epochs"]
        self.log_resources("Pre-Training")
        start_time = time.perf_counter()
        scaler = torch.amp.GradScaler('cuda')
        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        train(self.model, self.trainloader, optim, epochs, self.device)
        elapsed = max(0, time.perf_counter() - start_time)
        loss, accuracy = test(self.model, self.valloader, self.device)
        self.log_resources(f"Post-Training (took {elapsed:.2f}s)")
        return self.get_parameters({}), len(self.trainloader.dataset), {"accuracy": accuracy}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.log_resources("Pre-Evaluation")
        start_time = time.perf_counter()
        loss, accuracy = test(self.model, self.valloader, self.device)
        elapsed = max(0, time.perf_counter() - start_time)
        self.log_resources(f"Post-Evaluation (took {elapsed:.2f}s)")
        return float(loss), len(self.valloader.dataset), {"accuracy": accuracy}

# ... (same as before, just update server_address)
if __name__ == "__main__":
    cid = int(os.getenv("CLIENT_ID", 0))
    edge_id = 0 if cid in [0, 1] else 1  # Client 0,1 -> Edge 0; Client 2,3 -> Edge 1
    time.sleep(10)
    trainloaders, valloaders, _ = prepare_dataset(num_partitions=4, batch_size=16)  # 4 clients now
    client = FlowerClient(trainloaders[cid], valloaders[cid], num_classes=10, cid=cid)
    logger.info(f"Client {cid} starting...")
    fl.client.start_numpy_client(server_address=f"edge{edge_id}:8080", client=client)