from utilsP import *


# Main Script
if __name__ == "__main__":
    # Load datasets
    dataset_names, datasets_list = load_datasets()
    mnist_data = limit_dataset_size(datasets_list[0], max_size=10)
    cifar_data = limit_dataset_size(datasets_list[1], max_size=10)
    rotatedMNIST_data = limit_dataset_size(datasets_list[2], max_size=10)

    # Generate unit tasks (45 binary classification tasks)
    unit_tasks = generate_unit_tasks(dataset_names, datasets_list )

    # Generate 120 random task sequences
    random_task_sequences = generate_random_task_sequences(unit_tasks)

    # Generate 120 permuted task sequences from a fixed task set
    permuted_task_sequences = generate_permuted_task_sequences(unit_tasks)
    
    # Prepare MNIST and CIFAR-10 data for random task sequences
    #mnist_random_prepared = prepare_data_for_sequences(mnist_data, random_task_sequences['mnist'])
    #cifar_random_prepared = prepare_data_for_sequences(cifar_data, random_task_sequences['cifar10'])
    rotatedMNIST_random_prepared = prepare_data_for_sequences(mnist_data, random_task_sequences['rotatedMNIST'])

    # Prepare MNIST and CIFAR-10 data for permuted task sequences
    mnist_permuted_prepared = prepare_data_for_sequences(mnist_data, permuted_task_sequences['mnist'])
    #cifar_permuted_prepared = prepare_data_for_sequences(cifar_data, permuted_task_sequences['cifar10'])
    rotatedMNIST_permuted_prepared = prepare_data_for_sequences(mnist_data, permuted_task_sequences['rotatedMNIST'])
    print( permuted_task_sequences['mnist'])
    # Generate Task2Vec embeddings for MNIST unit tasks
    rotatedMNIST_embeddings = generate_task_embeddings(rotatedMNIST_permuted_prepared)
    mnist_embeddings = generate_task_embeddings(mnist_permuted_prepared)
    

    # Generate Task2Vec embeddings for CIFAR-10 unit tasks
    #cifar_embeddings = generate_task_embeddings(cifar_permuted_prepared)

    # Save embeddings
    with open('mnist_embeddings.p', 'wb') as f:
        pickle.dump(mnist_embeddings, f)
    
    #with open('cifar_embeddings.p', 'wb') as f:
        #pickle.dump(cifar_embeddings, f)

    with open('rotatedMNIST_embeddings.p', 'wb') as f:
        pickle.dump(mnist_embeddings, f)
    
    print("Embeddings for MNIST and CIFAR-10 unit tasks have been saved.")
