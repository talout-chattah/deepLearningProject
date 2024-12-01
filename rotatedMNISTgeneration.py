import gzip
import pickle
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
from PIL import Image

def generate_rotated_mnist():
    mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor())
    rotated_images, labels = [], []

    for img, label in mnist_data:
        for angle in range(0, 360, 45):  # Rotate every 45 degrees
            rotated_img = img.numpy().squeeze()  # Remove channel dim
            rotated_img = np.array(Image.fromarray(rotated_img * 255).rotate(angle)) / 255.0
            rotated_images.append(rotated_img)
            labels.append(label)

    # Save to .pkl.gz
    with gzip.open('rotatedMNIST.pkl.gz', 'wb') as f:
        pickle.dump((np.array(rotated_images), np.array(labels)), f)

generate_rotated_mnist()
