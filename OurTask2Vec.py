from utilsP import *
import tensorflow as tf
from torch import tensor
import random



def generateTaskEmbeddings(dataset_names, datasets):
    embeddings = []
    for name, dataset in zip(dataset_names, datasets):
        print(f"Embedding {name}")
        probe_network = get_model('resnet18', pretrained=True, num_classes=10)
        embeddings.append(Task2Vec(probe_network, max_samples=5000, skip_layers=6).embed(dataset))
    return embeddings


# Main Script
if __name__ == "__main__":
    # Load datasets
    dataset_names, datasets_list = load_datasets()
    mnist_data = limit_dataset_size(datasets_list[0], max_size=5000)

    rotatedMNISTrandom_data = []
    for item, label in mnist_data:
        itemrotated = torch.rot90(item, k=random.randint(0, 3), dims=(1, 2))
        rotatedMNISTrandom_data.append((itemrotated, label))
    
    rotatedMNIST90_data = []
    for item, label in mnist_data:
        itemrotated = torch.rot90(item, k=1, dims=(1, 2))
        rotatedMNIST90_data.append((itemrotated, label))
    
    rotatedMNIST180_data = []
    for item, label in mnist_data:
        itemrotated = torch.rot90(item, k=2, dims=(1, 2))
        rotatedMNIST180_data.append((itemrotated, label))
    
    rotatedMNIST270_data = []
    for item, label in mnist_data:
        itemrotated = torch.rot90(item, k=3, dims=(1, 2))
        rotatedMNIST270_data.append((itemrotated, label))


    data = (rotatedMNISTrandom_data,rotatedMNIST90_data,rotatedMNIST180_data,rotatedMNIST270_data,mnist_data)
    datasets_names=('rotatedMNISTrandom','rotatedMNIST90','rotatedMNIST180','rotatedMNIST270','MNIST')

    with tf.device('/GPU:0'):
        embeddings = generateTaskEmbeddings(datasets_names,data)
  
    
    # Save embeddings
    with open('yara_embeddings.p', 'wb') as f:
        pickle.dump((datasets_names, embeddings), f)

    print("yara Embeddings are ready!.")