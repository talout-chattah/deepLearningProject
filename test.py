import gzip
import pickle
from Task2vec import task_similarity

with open('saved_models/VCLBest.pkl', 'rb') as f:
            model = pickle.load(f, encoding='latin1')

print(type(model))

