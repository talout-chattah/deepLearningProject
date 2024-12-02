import itertools
import random
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist, cifar10
from Task2vec.task2vec import Task2Vec
from Task2vec.models import get_model
import Task2vec.datasets as datasets
import Task2vec.task_similarity
from itertools import combinations, permutations
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import gzip

# Step 1: Load MNIST and CIFAR-10 datasets
def load_datasets():
    dataset_names = ('mnist', 'cifar10', 'rotatedMNIST')
    
    # Change `root` with the directory you want to use to download the datasets
    dataset_list = [datasets.__dict__[name](root='./data')[0] for name in dataset_names[:2]]

    return dataset_names, dataset_list

# Step 2: Generate 45 unit tasks (all pairs of labels)
def generate_unit_tasks(dataset_names, dataset_list):
    # Number of labels for MNIST and CIFAR-10
    num_classes = {
        'mnist': 10,
        'cifar10': 10,
        'rotatedMNIST': 10
    }
    
    unit_tasks = {}
    
    for dataset_name, dataset in zip(dataset_names, dataset_list):
        # Generate all unique pairs of labels (0-9)
        label_pairs = list(combinations(range(num_classes[dataset_name]), 2))
        
        # For each pair of labels, create a unit task
        unit_tasks[dataset_name] = label_pairs

    return unit_tasks

# Step 3: Generate random task sequences
def generate_random_task_sequences(unit_tasks, num_sequences=120, sequence_length=5):
    task_sequences = {}
    
    for dataset_name, tasks in unit_tasks.items():
        task_sequences[dataset_name] = []
        
        for _ in range(num_sequences):
            # Randomly sample `sequence_length` tasks from the unit tasks
            sequence = random.sample(tasks, sequence_length)
            task_sequences[dataset_name].append(sequence)
    
    return task_sequences

# Step 4: Generate permuted task sequences from a fixed task set
def generate_permuted_task_sequences(unit_tasks, num_permutations=120):
    permuted_task_sequences = {}

    for dataset_name, tasks in unit_tasks.items():
        # Fix a subset of tasks (e.g., first 5 tasks)
        fixed_task_set = tasks[:5]
        
        # Generate all permutations of the fixed set
        all_permutations = list(permutations(fixed_task_set))
        
        # Randomly sample `num_permutations` permutations
        sampled_permutations = random.sample(all_permutations, min(num_permutations, len(all_permutations)))
        
        permuted_task_sequences[dataset_name] = sampled_permutations

    return permuted_task_sequences

# Step 5: Filter data for a specific task
class FilteredDataset(Dataset):
    """
    A custom dataset that filters the original dataset based on a label pair.
    
    Args:
        dataset: The original dataset to filter (e.g., MNIST or CIFAR-10).
        label_pair: A tuple of two labels to filter (e.g., (0, 1)).
    """
    def __init__(self, dataset, label_pair):
        if isinstance(label_pair, int):  # Single label
            label_pair = (label_pair,)
        
        self.dataset = dataset
        self.label_pair = label_pair
        self.filtered_indices = []
        
        # Filter the indices based on the label pair
        for i, (_, label) in enumerate(dataset):
            if label in self.label_pair:
                self.filtered_indices.append(i)
        
    def __len__(self):
        return len(self.filtered_indices)
    
    def get_dims(self):
        return 2, 10

    def __getitem__(self, idx):
        # Get the sample and label using the filtered index
        original_idx = self.filtered_indices[idx]
        sample, label = self.dataset[original_idx]
        return sample, label

# Step 6: Limit Dataset Size for Debugging (My PC contraints)
def limit_dataset_size(dataset, max_size=1000):
    """
    Limits the dataset size by selecting a random subset of the data.

    Args:
        dataset: The dataset to limit (e.g., MNIST or CIFAR-10).
        max_size: The maximum number of samples to retain in the dataset.

    Returns:
        A subset of the original dataset with at most `max_size` samples.
    """
    # Randomly select a subset of the dataset up to `max_size` samples
    if len(dataset) > max_size:
        indices = torch.randperm(len(dataset)).tolist()[:max_size]
        limited_data = torch.utils.data.Subset(dataset, indices)
    else:
        limited_data = dataset  # If dataset is smaller than max_size, keep all data

    return limited_data

# Step 7: Prepare data for all task sequences
def prepare_data_for_sequences(dataset, task_sequences):
    """
    Prepares filtered data for all task sequences.

    Args:
        dataset: The dataset to filter (e.g., MNIST or CIFAR-10).
        task_sequences: A list of task sequences, where each sequence is a list of label pairs.

    Returns:
        prepared_data: A list of lists, where each sublist contains filtered datasets
                       corresponding to the tasks in a sequence.
    """
    prepared_data = []

    for sequence in task_sequences:
        sequence_data = []
        for label_pair in sequence:
            if not isinstance(label_pair, tuple) or len(label_pair) != 2:
                raise ValueError(f"Invalid label pair: {label_pair}. Expected a tuple of two values.")
            
            filtered_data = FilteredDataset(dataset, label_pair)
            sequence_data.append(filtered_data)
        prepared_data.append(sequence_data)
    
    return prepared_data

# Step 8: Generate task embeddings for each unit task using Task2Vec
def generate_task_embeddings(mnist_permuted_prepared):
    embeddings = []
    for index, task in enumerate(mnist_permuted_prepared):  # Directly iterate over the datasets
        print(f"Task N°{index} is embedding")
        for subset in task:
            probe_network = get_model('resnet18', pretrained=True, num_classes=10).cuda()
            embeddings.append(Task2Vec(probe_network, max_samples=1000, skip_layers=6).embed(subset))
    return embeddings
