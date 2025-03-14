import flwr as fl
import torch
import psutil
import time
import logging
from collections import OrderedDict
from model import Net, train, test
from dataset import prepare_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] Client %(cid)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def log_resources(cid, phase):
    global net_io_start
    if "Pre" in phase:
        net_io_start = psutil.net_io_counters()
    cpu = psutil.cpu_percent(interval=0.1)
    net_io_end = psutil.net_io_counters()
    tx_mb = (net_io_end.bytes_sent - net_io_start.bytes_sent) / 1e6
    energy = cpu * 0.1 * 150 / 100
    logger.info(f"{phase} - CPU: {cpu:.1f}% | TX: {tx_mb:.2f} MB | Energy: {energy:.2f} J/s", extra={"cid": cid})

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, num_classes, cid):
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = Net(num_classes).to(device)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cid = cid

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        log_resources(self.cid, "Pre-Training")
        start_time = time.perf_counter()
        scaler = torch.amp.GradScaler('cuda')
        optim = torch.optim.SGD(self.model.parameters(), lr=config["lr"], momentum=config["momentum"])
        train(self.model, self.trainloader, optim, config["local_epochs"], self.device)
        elapsed = max(0, time.perf_counter() - start_time)
        loss, accuracy = test(self.model, self.valloader, self.device)
        log_resources(self.cid, f"Post-Training (took {elapsed:.2f}s)")
        return self.get_parameters({}), len(self.trainloader.dataset), {"accuracy": accuracy}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        log_resources(self.cid, "Pre-Evaluation")
        start_time = time.perf_counter()
        loss, accuracy = test(self.model, self.valloader, self.device)
        elapsed = max(0, time.perf_counter() - start_time)
        log_resources(self.cid, f"Post-Evaluation (took {elapsed:.2f}s)")
        return float(loss), len(self.valloader.dataset), {"accuracy": accuracy}

if __name__ == "__main__":
    import os
    cid = int(os.getenv("CLIENT_ID", 0))
    edge_id = 0 if cid in [0, 1] else 1
    try:
        trainloaders, valloaders, _ = prepare_dataset(num_partitions=4, batch_size=16)
        client = FlowerClient(trainloaders[cid], valloaders[cid], num_classes=10, cid=cid)
        logger.info(f"Connecting to Edge {edge_id} at edge{edge_id}:8080", extra={"cid": cid})
        time.sleep(10)
        fl.client.start_numpy_client(server_address=f"edge{edge_id}:8080", client=client)
    except Exception as e:
        logger.error(f"Failed: {str(e)}", extra={"cid": cid})
        raise