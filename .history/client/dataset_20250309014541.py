import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

def prepare_dataset(num_partitions=4, batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    num_samples = len(trainset)
    samples_per_partition = num_samples // num_partitions
    indices = torch.randperm(num_samples).tolist()
    trainloaders = []
    for i in range(num_partitions):
        start_idx = i * samples_per_partition
        end_idx = start_idx + samples_per_partition if i < num_partitions - 1 else num_samples
        subset_indices = indices[start_idx:end_idx]
        subset = Subset(trainset, subset_indices)
        trainloaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True))

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloaders, [], testloader