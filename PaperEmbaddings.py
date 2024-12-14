from utilsP import *
import tensorflow as tf
from torch import tensor
import random
# Main Script
if __name__ == "__main__":
    # Load datasets
    dataset_names, datasets_list = load_datasets()
    mnist_data = limit_dataset_size(datasets_list[0], max_size=1000)
    cifar_data = limit_dataset_size(datasets_list[1], max_size=1000)
    rotatedMNIST_data = []
    for item, label in mnist_data:
        itemrotated = torch.rot90(item, k=random.randint(0, 3), dims=(1, 2))
        rotatedMNIST_data.append((itemrotated,label))
    

    # Generate unit tasks (45 binary classification tasks)
    unit_tasks = generate_unit_tasks(dataset_names, datasets_list )

    # Generate 120 random task sequences
    random_task_sequences = generate_random_task_sequences(unit_tasks)

    # Generate 120 permuted task sequences from a fixed task set
    permuted_task_sequences = generate_permuted_task_sequences(unit_tasks)
    
    # Prepare MNIST and CIFAR-10 data for random task sequences
    mnist_random_prepared = prepare_data_for_sequences(mnist_data, random_task_sequences['mnist'])
    cifar_random_prepared = prepare_data_for_sequences(cifar_data, random_task_sequences['cifar10'])
    rotatedMNIST_random_prepared = prepare_data_for_sequences(rotatedMNIST_data, random_task_sequences['mnist'])


    # Generate Task2Vec embeddings for tasks
    rotatedMNIST_embeddings = generate_task_embeddings(rotatedMNIST_random_prepared)
    mnist_embeddings = generate_task_embeddings(mnist_random_prepared)
    cifar_embeddings = generate_task_embeddings(cifar_random_prepared)

    # Save embeddings
    with open('mnist_embeddings.p', 'wb') as f:
        pickle.dump((random_task_sequences['mnist'],mnist_embeddings), f)
    
    with open('cifar_embeddings.p', 'wb') as f:
        pickle.dump(cifar_embeddings, f)

    with open('rotatedMNIST_embeddings.p', 'wb') as f:
        pickle.dump((random_task_sequences['mnist'],rotatedMNIST_embeddings), f)
    
    print("Embeddings for MNIST and rotatedMnist unit tasks have been saved.")
