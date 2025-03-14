import flwr as fl
import torch
import logging
from flwr.common import ndarrays_to_parameters
from model import Net, test
from dataset import prepare_dataset

logger = logging.getLogger("cloud")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(fmt="%(asctime)s [%(levelname)s] Cloud - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logger.handlers = [handler]

def evaluate_fn(server_round, parameters, config):
    model = Net(num_classes=10)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    _, _, testloader = prepare_dataset(num_partitions=4, batch_size=128)
    state_dict = {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)}
    model.load_state_dict(state_dict, strict=True)
    loss, accuracy = test(model, testloader, device)
    return loss, {"accuracy": accuracy}

def fit_config(server_round: int):
    return {"local_epochs": 1}

def fit_metrics_aggregation_fn(fit_metrics):
    logger.info(f"Raw fit_metrics received: {fit_metrics}", extra={"server": "cloud"})
    total_examples = sum([n for n, _ in fit_metrics])
    avg_loss = sum([m.get("loss", 0.0) * n for n, m in fit_metrics]) / total_examples if total_examples > 0 else 0.0
    avg_accuracy = sum([m.get("accuracy", 0.0) * n for n, m in fit_metrics]) / total_examples if total_examples > 0 else 0.0
    return {"loss": avg_loss, "accuracy": avg_accuracy}

def evaluate_metrics_aggregation_fn(eval_metrics):
    total_examples = sum([n for n, _ in eval_metrics])
    avg_accuracy = sum([m["accuracy"] * n for n, m in eval_metrics]) / total_examples if total_examples > 0 else 0.0
    return {"accuracy": avg_accuracy}

if __name__ == "__main__":
    model = Net(num_classes=10)
    initial_params = [val.cpu().numpy() for val in model.state_dict().values()]
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=evaluate_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        on_fit_config_fn=fit_config,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        initial_parameters=ndarrays_to_parameters(initial_params),
    )
    try:
        logger.info("Starting Cloud Server on 0.0.0.0:8080")
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=10),
            strategy=strategy,
            grpc_max_message_length=512 * 1024 * 1024,
        )
    except Exception as e:
        logger.error(f"Cloud Server failed: {str(e)}", extra={"server": "cloud"})
        raise