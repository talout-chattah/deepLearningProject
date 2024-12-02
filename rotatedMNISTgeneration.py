import torch
import pickle
import gzip
from torchvision import datasets, transforms
from torch.utils.data import Dataset

class RotateMNIST(Dataset):
    def __init__(self, root, train=True, transform=None, download=True):
        self.mnist = datasets.MNIST(root=root, train=train, download=download)
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),  # Ensures the image stays grayscale
            transforms.RandomRotation(degrees=(0, 360)),
        ])
    
    def __len__(self):
        return len(self.mnist)
    
    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

def save_dataset_to_file(dataset, file_path):
    example_tensor, example_label = dataset[0]
    print((example_tensor.shape, example_label))
    with gzip.open(file_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved to {file_path}")

# Create the rotated MNIST dataset
dataset = RotateMNIST(root="./data", train=True, download=True)

# Save the dataset to a .pkl.gz file
save_dataset_to_file(dataset, './data/rotatedMNIST.pkl.gz')
