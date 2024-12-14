import numpy as np
from scipy.ndimage import rotate
import tensorflow.compat.v1 as tf
import gzip
import pickle
import sys
sys.path.extend(['alg/'])
from VCL.ddm.alg import vcl, coreset, utils
from copy import deepcopy
from utilsP import *
import matplotlib.pyplot as plt
import seaborn as sns

tf.disable_v2_behavior()

class YaraGenerator:
    def __init__(self, dataset_paths, max_samples=None):
        """
        Initializes the generator with paths to datasets for each task and an optional sample limit.

        Parameters:
        - dataset_paths: List of file paths, where each path corresponds to a rotated MNIST dataset.
        - max_samples: Maximum number of samples to use from each dataset (None for no limit).
        """
        self.dataset_paths = dataset_paths
        self.max_samples = max_samples
        self.max_iter = len(dataset_paths)
        self.cur_iter = 0

    def get_dims(self):
        """
        Gets input and output dimensions for the dataset.
        Assumes all datasets share the same dimensions.
        """
        # Load the first dataset to infer dimensions
        with gzip.open(self.dataset_paths[0], 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        input_dim = train_set[0].shape[1]  # Number of features (e.g., 784 for MNIST)
        output_dim = 10  # Number of classes
        return input_dim, output_dim

    def limit_samples(self, x, y):
        """
        Limits the number of samples in the dataset if max_samples is specified.
        """
        if self.max_samples is not None and len(x) > self.max_samples:
            indices = np.random.choice(len(x), self.max_samples, replace=False)
            return x[indices], y[indices]
        return x, y

    def next_task(self):
        """
        Loads the next rotated dataset and applies the sample limit if specified.
        """
        if self.cur_iter >= self.max_iter:
            raise Exception("Number of tasks exceeded!")

        # Load the dataset for the current task
        dataset_path = self.dataset_paths[self.cur_iter]
        with gzip.open(dataset_path, 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

        # Combine training and validation sets
        x_train = np.vstack((train_set[0], valid_set[0]))
        y_train = np.hstack((train_set[1], valid_set[1]))
        x_test = test_set[0]
        y_test = test_set[1]

        # Apply sample limit
        x_train, y_train = self.limit_samples(x_train, y_train)
        x_test, y_test = self.limit_samples(x_test, y_test)

        # One-hot encode labels
        y_train = np.eye(10)[y_train]
        y_test = np.eye(10)[y_test]

        self.cur_iter += 1

        return x_train, y_train, x_test, y_test



dataset_paths = [
    "data/mnist.pkl.gz",
    "data/rotatedmnist45.pkl.gz",
    "data/rotatedmnist90.pkl.gz",
    "data/rotatedmnist135.pkl.gz",
    "data/rotatedmnist180.pkl.gz",
    "data/rotatedmnist225.pkl.gz",
    "data/rotatedmnist270.pkl.gz",
    "data/rotatedmnist315.pkl.gz",
]

hidden_size = [256, 256]
batch_size = 64
no_epochs = 120 # number of epochs
single_head = True

# Run vanilla VCL
print("vanilla VCL:")

tf.set_random_seed(12)
np.random.seed(1)
coreset_size = 0 

task_gen = YaraGenerator(dataset_paths, max_samples= 5000)

with tf.device('/GPU:0'):
    rand_vcl_result = vcl.run_vcl(hidden_size, no_epochs, task_gen, 
                                  coreset.rand_from_batch, coreset_size, batch_size, single_head, "saved_models/VCL.pkl")
    print (rand_vcl_result)

# Accuracy matrix
accuracy_matrix = rand_vcl_result

# Save the accuracy matrix
with open("Accuracy_Matrix/VCL_accuracy_matrix.pkl", "wb") as f:
    pickle.dump(accuracy_matrix, f)