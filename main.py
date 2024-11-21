import itertools
import random
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist, cifar10
from Task2vec.task2vec import Task2Vec
from Task2vec.models import get_model
import Task2vec.datasets
import Task2vec.task_similarity

# Step 1: Load MNIST and CIFAR-10 datasets
def load_datasets():
    # Load MNIST
    (mnist_train_x, mnist_train_y), (mnist_test_x, mnist_test_y) = mnist.load_data()
    mnist_train_x = mnist_train_x.astype('float32').reshape(-1, 28, 28, 1) / 255.0
    mnist_test_x = mnist_test_x.astype('float32').reshape(-1, 28, 28, 1) / 255.0

    # Load CIFAR-10
    (cifar_train_x, cifar_train_y), (cifar_test_x, cifar_test_y) = cifar10.load_data()
    cifar_train_y = cifar_train_y.flatten()
    cifar_test_y = cifar_test_y.flatten()
    cifar_train_x = cifar_train_x.astype('float16') / 255.0
    cifar_test_x = cifar_test_x.astype('float16') / 255.0

    return {
        "mnist": (mnist_train_x, mnist_train_y, mnist_test_x, mnist_test_y),
        "cifar10": (cifar_train_x, cifar_train_y, cifar_test_x, cifar_test_y),
    }

# Step 2: Generate 45 unit tasks (all pairs of labels)
def generate_unit_tasks(labels):
    return list(itertools.combinations(labels, 2))  # All binary classification tasks

# Step 3: Generate random task sequences
def generate_random_task_sequences(unit_tasks, num_sequences=120, sequence_length=5):
    return [
        random.sample(unit_tasks, sequence_length) for _ in range(num_sequences)
    ]

# Step 4: Generate permuted task sequences from a fixed task set
def generate_permuted_task_sequences(unit_tasks, num_permutations=120):
    fixed_task_set = random.sample(unit_tasks, 5)  # Fixed set of 5 tasks
    permutations = list(itertools.permutations(fixed_task_set))
    return permutations[:num_permutations]  # Limit to 120 permutations

# Step 5: Filter data for a specific task
def filter_data_by_task(x, y, task):
    label_a, label_b = task
    mask = (y == label_a) | (y == label_b)
    x_filtered, y_filtered = x[mask], y[mask]
    y_filtered = (y_filtered == label_b).astype(int)  # Binary labels (0 or 1)
    return x_filtered, y_filtered

# Step 6: Limit Dataset Size for Debugging (My PC contraints)
def limit_dataset(dataset, train_size=10000, test_size=2000):
    train_x, train_y, test_x, test_y = dataset
    train_x, train_y = train_x[:train_size], train_y[:train_size]
    test_x, test_y = test_x[:test_size], test_y[:test_size]
    return train_x, train_y, test_x, test_y

# Step 7: Prepare data for all task sequences
def prepare_data_for_sequences(dataset, task_sequences):
    train_x, train_y, test_x, test_y = dataset
    prepared_sequences = []

    for sequence in task_sequences:
        sequence_data = []
        for task in sequence:
            # Filter data for current task
            task_train_x, task_train_y = filter_data_by_task(train_x, train_y, task)
            task_test_x, task_test_y = filter_data_by_task(test_x, test_y, task)

            # Split train into training and validation
            task_train_x, task_val_x, task_train_y, task_val_y = train_test_split(
                task_train_x, task_train_y, test_size=0.2, random_state=42
            )

            # Collect data for the task
            sequence_data.append({
                "train": (task_train_x, task_train_y),
                "val": (task_val_x, task_val_y),
                "test": (task_test_x, task_test_y),
            })
        prepared_sequences.append(sequence_data)
    return prepared_sequences

# Step 8: Generate task embeddings for each unit task using Task2Vec
def generate_task_embeddings(dataset, permuted_unit_tasks):
    train_x, train_y, test_x, test_y = dataset
    embeddings = []

    for sequence in permuted_unit_tasks:
        for task in sequence:
        # Unpack data for this task
            task_train_x, task_train_y = task["train"]
            task_val_x, task_val_y = task["val"]
            task_test_x, task_test_y = task["test"]

            probe_network = get_model('resnet18', pretrained=True, num_classes=2)

            # Save the embedding for the current task
            embeddings.append(Task2Vec(probe_network, max_samples=1000, skip_layers=6).embed(task_train_x)).cuda() 

    return embeddings

# Main Script
if __name__ == "__main__":
    # Load datasets
    datasets = load_datasets()
    mnist_data = datasets["mnist"]
    cifar_data = datasets["cifar10"]

    # Limit dataset sizes for testing (optional, can be removed for full dataset processing)
    cifar_data = limit_dataset(cifar_data)

    # Generate unit tasks (45 binary classification tasks)
    labels = list(range(10))  # Labels are 0 to 9
    unit_tasks = generate_unit_tasks(labels)

    # Generate 120 random task sequences
    random_task_sequences = generate_random_task_sequences(unit_tasks)

    # Generate 120 permuted task sequences from a fixed task set
    permuted_task_sequences = generate_permuted_task_sequences(unit_tasks)

    # Prepare MNIST and CIFAR-10 data for random task sequences
    mnist_random_prepared = prepare_data_for_sequences(mnist_data, random_task_sequences)
    cifar_random_prepared = prepare_data_for_sequences(cifar_data, random_task_sequences)
    

    # Prepare MNIST and CIFAR-10 data for permuted task sequences
    mnist_permuted_prepared = prepare_data_for_sequences(mnist_data, permuted_task_sequences)
    cifar_permuted_prepared = prepare_data_for_sequences(cifar_data, permuted_task_sequences)
    #print(mnist_permuted_prepared)
    # Generate Task2Vec embeddings for MNIST unit tasks
    mnist_embeddings = generate_task_embeddings(mnist_data, mnist_permuted_prepared)

    # Generate Task2Vec embeddings for CIFAR-10 unit tasks
    cifar_embeddings = generate_task_embeddings(cifar_data, cifar_permuted_prepared)

    # Save embeddings
    with open('mnist_embeddings.p', 'wb') as f:
        pickle.dump(mnist_embeddings, f)
    
    with open('cifar_embeddings.p', 'wb') as f:
        pickle.dump(cifar_embeddings, f)
    
    print("Embeddings for MNIST and CIFAR-10 unit tasks have been saved.")

    """
    # Example: Display the shape of data for the first task in MNIST (random)
    example_task = mnist_random_prepared[0][0]
    print("MNIST Random Task 1 - Train Data Shape:", example_task["train"][0].shape)
    print("MNIST Random Task 1 - Train Labels Shape:", example_task["train"][1].shape)

    # Example: Display the shape of data for the first task in CIFAR-10 (permuted)
    example_task_cifar = cifar_permuted_prepared[0][0]
    print("CIFAR-10 Permuted Task 1 - Train Data Shape:", example_task_cifar["train"][0].shape)
    print("CIFAR-10 Permuted Task 1 - Train Labels Shape:", example_task_cifar["train"][1].shape)
    """
