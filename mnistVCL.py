from utilsP import *
from VCL.ddm.alg import vcl
import VCL.ddm.alg.coreset
import tensorflow.compat.v1 as tf


tf.disable_v2_behavior()


# Load datasets
dataset_names, datasets_list = load_datasets()
mnist_data = limit_dataset_size(datasets_list[0], max_size=10)

# Generate unit tasks (45 binary classification tasks)
unit_tasks = generate_unit_tasks(dataset_names, datasets_list )

# Generate 120 random task sequences
random_task_sequences = generate_random_task_sequences(unit_tasks)

# Generate 120 permuted task sequences from a fixed task set
permuted_task_sequences = generate_permuted_task_sequences(unit_tasks)

# Prepare MNIST data for random task sequences
mnist_random_prepared = prepare_data_for_sequences(mnist_data, random_task_sequences['mnist'])

# Prepare MNIST data for permuted task sequences
mnist_permuted_prepared = prepare_data_for_sequences(mnist_data, permuted_task_sequences['mnist'])



hidden_size = [100, 100]
batch_size = 256
no_epochs = 10 # number of epochs
single_head = True
num_tasks = 5
tf.set_random_seed(12)
np.random.seed(1)
coreset_size = 0
vcl_result = vcl.run_vcl(hidden_size, no_epochs, mnist_permuted_prepared, coreset.rand_from_batch, coreset_size, batch_size, single_head)
print (vcl_result)

