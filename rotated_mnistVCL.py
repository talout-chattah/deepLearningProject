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

class TaskDefinedRotatedMnistGenerator:
    def __init__(self, task_list, max_samples=None):
        # Load MNIST dataset
        with gzip.open('data/rotatedmnist.pkl.gz', 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.Y_train = np.hstack((train_set[1], valid_set[1]))
        self.X_test = test_set[0]
        self.Y_test = test_set[1]

        # Store task list and initialize task index
        self.task_list = task_list
        self.max_iter = len(task_list)
        self.cur_iter = 0
        self.max_samples = max_samples  # Limit for the number of samples per class

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], 10

    def _limit_data_size(self, x_data, y_data, selected_classes):
        """Limit the size of the data per class if max_samples is specified."""
        if self.max_samples is None:
            return x_data, y_data

        limited_x = []
        limited_y = []
        for cls in selected_classes:
            class_mask = y_data == cls
            x_cls = x_data[class_mask]
            y_cls = y_data[class_mask]
            
            # Limit samples for the class
            if x_cls.shape[0] > self.max_samples:
                indices = np.random.choice(x_cls.shape[0], self.max_samples, replace=False)
                x_cls = x_cls[indices]
                y_cls = y_cls[indices]

            limited_x.append(x_cls)
            limited_y.append(y_cls)

        # Concatenate limited data
        return np.vstack(limited_x), np.hstack(limited_y)

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception("Number of tasks exceeded!")
        
        # Get current task's class pairs
        class_pairs = self.task_list[self.cur_iter]
        selected_classes = [c for pair in class_pairs for c in pair]

        # Filter training data for the selected classes
        train_mask = np.isin(self.Y_train, selected_classes)
        x_train = self.X_train[train_mask]
        y_train = self.Y_train[train_mask]
        x_train, y_train = self._limit_data_size(x_train, y_train, selected_classes)
        y_train_one_hot = np.eye(10)[y_train]

        # Filter test data for the selected classes
        test_mask = np.isin(self.Y_test, selected_classes)
        x_test = self.X_test[test_mask]
        y_test = self.Y_test[test_mask]
        x_test, y_test = self._limit_data_size(x_test, y_test, selected_classes)
        y_test_one_hot = np.eye(10)[y_test]

        # Move to the next task
        self.cur_iter += 1

        return x_train, y_train_one_hot, x_test, y_test_one_hot


hidden_size = [256, 256]
batch_size = None #64
no_epochs = 120 # number of epochs
single_head = True

# Run vanilla VCL
print("vanilla VCL:")

tf.set_random_seed(12)
np.random.seed(1)
coreset_size = 0

dataset_names, datasets_list = load_datasets()
# Generate unit tasks (45 binary classification tasks)
unit_tasks = generate_unit_tasks(dataset_names, datasets_list )


# Generate 120 permuted task sequences from a fixed task set
permuted_task_sequences = generate_permuted_task_sequences(unit_tasks) 

numtasks = 10
tasks = []
for i in range(numtasks):
    tasks.append(permuted_task_sequences['mnist'][i]) 

task_gen = TaskDefinedRotatedMnistGenerator(permuted_task_sequences['mnist'], max_samples=500)

with tf.device('/GPU:0'):
    rand_vcl_result = vcl.run_vcl(hidden_size, no_epochs, task_gen, 
                                  coreset.rand_from_batch, coreset_size, batch_size, single_head)
    print (rand_vcl_result)

# Accuracy matrix
accuracy_matrix = rand_vcl_result

# Save the accuracy matrix
with open("accuracy_matrix120.pkl", "wb") as f:
    pickle.dump(accuracy_matrix, f)