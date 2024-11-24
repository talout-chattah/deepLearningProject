from utilsP import *


# Main Script
if __name__ == "__main__":
    # Load datasets
    dataset_names, datasets_list = load_datasets()
    mnist_data = limit_dataset_size(datasets_list[0], max_size=1000)
    cifar_data = limit_dataset_size(datasets_list[1], max_size=1000)

    # Generate unit tasks (45 binary classification tasks)
    unit_tasks = generate_unit_tasks(dataset_names, datasets_list )

    # Generate 120 random task sequences
    random_task_sequences = generate_random_task_sequences(unit_tasks)

    # Generate 120 permuted task sequences from a fixed task set
    permuted_task_sequences = generate_permuted_task_sequences(unit_tasks)
    
    # Prepare MNIST and CIFAR-10 data for random task sequences
    mnist_random_prepared = prepare_data_for_sequences(mnist_data, random_task_sequences['mnist'])
    #cifar_random_prepared = prepare_data_for_sequences(cifar_data, random_task_sequences['cifar10'])
    

    # Prepare MNIST and CIFAR-10 data for permuted task sequences
    mnist_permuted_prepared = prepare_data_for_sequences(mnist_data, permuted_task_sequences['mnist'])
    #cifar_permuted_prepared = prepare_data_for_sequences(cifar_data, permuted_task_sequences['cifar10'])
    #print(mnist_permuted_prepared)
    # Generate Task2Vec embeddings for MNIST unit tasks
    mnist_embeddings = generate_task_embeddings(mnist_permuted_prepared)

    # Generate Task2Vec embeddings for CIFAR-10 unit tasks
    #cifar_embeddings = generate_task_embeddings(cifar_permuted_prepared)

    # Save embeddings
    with open('mnist_embeddings.p', 'wb') as f:
        pickle.dump(mnist_embeddings, f)
    
    #with open('cifar_embeddings.p', 'wb') as f:
        #pickle.dump(cifar_embeddings, f)
    
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
