from utilsP import *
import tensorflow as tf
from torch import tensor
from torchvision.transforms import functional as F
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
    mnist_data = limit_dataset_size(datasets_list[0], max_size=3000)

    rotatedMNISTrandom_data = []
    for item, label in mnist_data:
        itemrotated = F.rotate(item, random.choice([0, 45 ,90, 135, 180, 225, 270, 315]), interpolation=F.InterpolationMode.BILINEAR)
        rotatedMNISTrandom_data.append((itemrotated, label))
    
    rotatedMNIS45_data = []
    for item, label in mnist_data:
        itemrotated = F.rotate(item, 45, interpolation=F.InterpolationMode.BILINEAR)
        rotatedMNIS45_data.append((itemrotated, label))
    
    rotatedMNIST90_data = []
    for item, label in mnist_data:
        itemrotated = torch.rot90(item, k=1, dims=(1, 2))
        rotatedMNIST90_data.append((itemrotated, label))
    
    rotatedMNIS135_data = []
    for item, label in mnist_data:
        itemrotated = F.rotate(item, 135, interpolation=F.InterpolationMode.BILINEAR)
        rotatedMNIS135_data.append((itemrotated, label))
    
    rotatedMNIST180_data = []
    for item, label in mnist_data:
        itemrotated = torch.rot90(item, k=2, dims=(1, 2))
        rotatedMNIST180_data.append((itemrotated, label))
    
    rotatedMNIS225_data = []
    for item, label in mnist_data:
        itemrotated = F.rotate(item, 225, interpolation=F.InterpolationMode.BILINEAR)
        rotatedMNIS225_data.append((itemrotated, label))
    
    rotatedMNIST270_data = []
    for item, label in mnist_data:
        itemrotated = torch.rot90(item, k=3, dims=(1, 2))
        rotatedMNIST270_data.append((itemrotated, label))

    rotatedMNIS315_data = []
    for item, label in mnist_data:
        itemrotated = F.rotate(item, 315, interpolation=F.InterpolationMode.BILINEAR)
        rotatedMNIS315_data.append((itemrotated, label))
    
    data = (mnist_data,rotatedMNIS45_data,rotatedMNIST90_data,rotatedMNIS135_data,rotatedMNIST180_data,rotatedMNIS225_data,rotatedMNIST270_data, rotatedMNIS315_data, rotatedMNISTrandom_data)
    datasets_names=('MNIST','rotatedMNIST45','rotatedMNIST90','rotatedMNIS135','rotatedMNIST180', 'rotatedMNIS225', 'rotatedMNIST270', 'rotatedMNIS315','rotatedMNISTrandom')

    with tf.device('/GPU:0'):
        embeddings = generateTaskEmbeddings(datasets_names,data)
  
    
    # Save embeddings
    with open('Hamidi_embeddings.p', 'wb') as f:
        pickle.dump((datasets_names, embeddings), f)

    print("Hamidi Embeddings are ready!.")