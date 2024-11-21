import numpy as np
import tensorflow.compat.v1 as tf
import gzip
import pickle
import sys
sys.path.extend(['alg/'])
import vcl
import coreset
import utils
from copy import deepcopy
import time
import psutil

tf.disable_v2_behavior()

class PermutedMnistGenerator():
    def __init__(self, max_iter=10):
        with gzip.open('data/mnist.pkl.gz', 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        """f = gzip.open('data/mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = pickle.load(f)"""
        f.close()

        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.Y_train = np.hstack((train_set[1], valid_set[1]))
        self.X_test = test_set[0]
        self.Y_test = test_set[1]
        self.max_iter = max_iter
        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], 10

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            np.random.seed(self.cur_iter)
            perm_inds = list(range(self.X_train.shape[1]))
            np.random.shuffle(perm_inds)

            # Retrieve train data
            next_x_train = deepcopy(self.X_train)
            next_x_train = next_x_train[:,perm_inds]
            next_y_train = np.eye(10)[self.Y_train]

            # Retrieve test data
            next_x_test = deepcopy(self.X_test)
            next_x_test = next_x_test[:,perm_inds]
            next_y_test = np.eye(10)[self.Y_test]

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

hidden_size = [100, 100]
batch_size = 256
no_epochs = 10 # number of epochs
single_head = True
num_tasks = 5

# Run vanilla VCL
print("vanilla VCL:")
# Measure initial CPU and memory usage
process = psutil.Process()
initial_cpu = psutil.cpu_percent(interval=1)
initial_memory = process.memory_info().rss / (1024 * 1024)  # in MB

# Record the start time
start_time = time.time()

tf.set_random_seed(12)
np.random.seed(1)

coreset_size = 0
data_gen = PermutedMnistGenerator(num_tasks)
vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen, coreset.rand_from_batch, coreset_size, batch_size, single_head)
print (vcl_result)

# Record the end time
end_time = time.time()

# Measure final CPU and memory usage
final_cpu = psutil.cpu_percent(interval=1)
final_memory = process.memory_info().rss / (1024 * 1024)  # in MB

# Calculate the time taken
time_taken = end_time - start_time

# Calculate the CPU and memory usage
cpu_usage = final_cpu - initial_cpu
memory_usage = final_memory - initial_memory


print(f"Code for vanilla VCL executed in {time_taken:.4f} seconds.")
print(f"CPU usage: {cpu_usage:.2f}%")
print(f"Memory usage: {memory_usage:.2f} MB")


# Run random coreset VCL
print("random coreset VCL:")

# Measure initial CPU and memory usage
process = psutil.Process()
initial_cpu = psutil.cpu_percent(interval=1)
initial_memory = process.memory_info().rss / (1024 * 1024)  # in MB

# Record the start time
start_time = time.time()

tf.reset_default_graph()
tf.set_random_seed(12)
np.random.seed(1)

coreset_size = 200
data_gen = PermutedMnistGenerator(num_tasks)
rand_vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen, 
    coreset.rand_from_batch, coreset_size, batch_size, single_head)
print (rand_vcl_result)

# Record the end time
end_time = time.time()

# Measure final CPU and memory usage
final_cpu = psutil.cpu_percent(interval=1)
final_memory = process.memory_info().rss / (1024 * 1024)  # in MB

# Calculate the time taken
time_taken = end_time - start_time

# Calculate the CPU and memory usage
cpu_usage = final_cpu - initial_cpu
memory_usage = final_memory - initial_memory


print(f"Code for random coreset VCL executed in {time_taken:.4f} seconds.")
print(f"CPU usage: {cpu_usage:.2f}%")
print(f"Memory usage: {memory_usage:.2f} MB")



# Run k-center coreset VCL
tf.reset_default_graph()
tf.set_random_seed(12)
np.random.seed(1)

data_gen = PermutedMnistGenerator(num_tasks)
kcen_vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen, 
    coreset.k_center, coreset_size, batch_size, single_head)
print (kcen_vcl_result)

# Plot average accuracy
vcl_avg = np.nanmean(vcl_result, 1)
rand_vcl_avg = np.nanmean(rand_vcl_result, 1)
kcen_vcl_avg = np.nanmean(kcen_vcl_result, 1)
utils.plot('results/permuted.jpg', vcl_avg, rand_vcl_avg, kcen_vcl_avg)
