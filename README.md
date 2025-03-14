# Federated Learning System with Hierarchical Aggregation

This project implements a hierarchical federated learning (FL) system using the Flower framework. It trains a convolutional neural network (CNN) on the MNIST dataset across three levels: clients, edges, and a central cloud server. The goal is to demonstrate distributed machine learning with privacy-preserving aggregation, where local client data never leaves its device, and intermediate edge nodes reduce communication overhead to the cloud.

## System Overview

The system is structured in three tiers:

1. **Clients** (4 total):
   - Train a local CNN model on a partition of the MNIST dataset.
   - Send updated weights and metrics (loss, accuracy) to their assigned edge node after local training.
   - Two clients per edge (e.g., Clients 0 & 2 on Edge 0, Clients 1 & 3 on Edge 1).

2. **Edges** (2 total):
   - Act as mini FL servers for their clients, aggregating weights and metrics over 10 rounds using FedAvg.
   - Act as clients to the cloud, sending aggregated results after their local rounds.
   - Reduce communication load by pre-aggregating client updates.

3. **Cloud** (1 server):
   - Coordinates the global FL process, aggregating edge results over 30 rounds.
   - Updates and distributes the global model weights back to edges.

### Data Flow
- **Clients → Edges**: Clients train locally (1 epoch default), send weights and metrics to their edge.
- **Edges → Cloud**: Edges aggregate client results over 10 rounds, send aggregated weights and metrics to the cloud.
- **Cloud → Edges**: Cloud aggregates edge results, updates the global model, and sends new weights back to edges for the next round.

### Model
- **Architecture**: A simple CNN (`Net`) from `model.py`, designed for MNIST digit classification (10 classes).
- **Dataset**: MNIST, split into 4 partitions (one per client, ~15,000 examples each).

## Prerequisites

- **Docker**: For containerized deployment.
- **Docker Compose**: To orchestrate the cloud, edges, and clients.
- **Python 3.9+**: If running locally (Flower and dependencies).
- **Dependencies**: Listed in `requirements.txt` (e.g., `flwr`, `torch`, `psutil`).

## Project Structure
```bash
├── client.py         # Client logic: local training and metrics reporting
├── edge.py           # Edge logic: aggregates clients, communicates with cloud
├── server.py         # Cloud logic: global aggregation and coordination
├── model.py          # CNN model definition (Net)
├── dataset.py        # MNIST dataset loading and partitioning
├── docker-compose.yml # Docker setup for cloud, edges, and clients
├── requirements.txt  # Python dependencies
└── README.md         # This file
```


## Setup

### Using Docker Compose
1. **Clone the Repository**:
   ```bash
   git clone <repo-url>
   cd <repo-name>
   ```

2. **Build and Run**:
   ```bash
   docker-compose up --build
   ```
- Starts 1 cloud (cloud-1), 2 edges (edge0-1-1, edge1-1-1), and 4 clients (client0-1-1, etc.).
- Containers connect via hostnames (e.g., cloud-1:8080, edge0-1-1:8080).

## Usage
**Training** :
- Launch the system with docker-compose up --build.
- Clients train on their MNIST partitions (1 epoch per round).
- Edges aggregate over 10 rounds, then send results to the cloud.
- Cloud runs 30 rounds, aggregating edge results into a global model.
  
**Monitoring**:

 Logs show progress:
- Clients: Training loss and accuracy and ressources usage.
- Edges: Aggregated metrics per round.
- Cloud: Global aggregation and evaluation.
  
**Results**:
- Expected client accuracies: ~96-97% after 10 edge rounds.
- Expected global accuracy: ~97-98% after cloud aggregation.

