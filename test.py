import gzip
import pickle
from Task2vec import task_similarity

with open('saved_models/VCLBest.pkl', 'rb') as f:
            model = pickle.load(f, encoding='latin1')

print(type(model))

model = MFVI_NN(
    input_size=model['in_dim'],
    hidden_size=model['hidden_size'],
    output_size=model['out_dim'],
    training_size=0,  # Not training
    prev_means=model['weights'][0],
    prev_log_variances=model['weights'][1],
    learning_rate=0.001
)
