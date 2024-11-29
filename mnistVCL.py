import numpy as np
import tensorflow.compat.v1 as tf
import gzip
import pickle
import sys
sys.path.extend(['alg/'])
from VCL.ddm.alg import vcl, coreset, utils
from copy import deepcopy
from utilsP import *

tf.disable_v2_behavior()

class MnistTaskGenerator:
    def __init__(self, tasks, dataset_path='data/mnist.pkl.gz'):
        """
        Args:
            tasks: List of tuples representing the tasks. Each tuple contains two digits (e.g., (0, 1)).
            dataset_path: Path to the MNIST dataset file.
        """
        with gzip.open(dataset_path, 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.Y_train = np.hstack((train_set[1], valid_set[1]))
        self.X_test = test_set[0]
        self.Y_test = test_set[1]
        self.tasks = tasks
        self.max_iter = len(self.tasks)
        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], 10

    def next_task(self):
        if self.cur_iter >= len(self.tasks):
            raise Exception('Number of tasks exceeded!')

        # Get the current task digit pair
        task_digits = self.tasks[self.cur_iter]
        task_filter_train = np.isin(self.Y_train, task_digits)
        task_filter_test = np.isin(self.Y_test, task_digits)

        # Retrieve filtered train data
        next_x_train = self.X_train[task_filter_train]
        next_y_train = np.eye(10)[self.Y_train[task_filter_train]]

        # Retrieve filtered test data
        next_x_test = self.X_test[task_filter_test]
        next_y_test = np.eye(10)[self.Y_test[task_filter_test]]

        self.cur_iter += 1

        return next_x_train, next_y_train, next_x_test, next_y_test

hidden_size = [100, 100]
batch_size = 256
no_epochs = 10 # number of epochs
single_head = True
num_tasks = 5

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

print( permuted_task_sequences['mnist'])

for task in permuted_task_sequences['mnist']:
    # Initialize the generator
    task_gen = MnistTaskGenerator(task)

    vcl_result = vcl.run_vcl(hidden_size, no_epochs, task_gen, coreset.rand_from_batch, coreset_size, batch_size, single_head)
    print (vcl_result)
