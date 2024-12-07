import gzip
import pickle
import tensorflow as tf
with gzip.open('data/mnist.pkl.gz', 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

with gzip.open('data/rotatedmnist.pkl.gz', 'rb') as f:
            rotated_train_set, rotated_valid_set, rotated_test_set = pickle.load(f, encoding='latin1')


with tf.device('/GPU:0'):
    print("train_set",train_set)
    print("rotated_train_set",rotated_train_set)