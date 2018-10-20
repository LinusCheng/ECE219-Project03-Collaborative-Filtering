import os
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise.similarities import pearson
import matplotlib.pyplot as plt
import numpy as np

file_path = os.path.expanduser('ml-latest-small/ratings_new.csv')
reader = Reader(sep=',')
data = Dataset.load_from_file(file_path, reader=reader)

sim_options = {'name': 'pearson',
              'user_based': True
              }



"""=== Q30 ==="""

print("==== Q30 ====")


avg_rmse = []
avg_mae = []
all_k = []
prev_rmse = 2
conv_k = 0




algo = KNNWithMeans(k=i, sim_options=sim_options)
  
  
  
output = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=10,  verbose=True, n_jobs=1)
avg_rmse.append(np.mean(output['test_rmse']))
avg_mae.append(np.mean(output['test_mae']))


if np.mean(output['test_rmse']) > 0 and prev_rmse-np.mean(output['test_rmse']) < 0.0005:
    conv_k = i*2
prev_rmse = np.mean(output['test_rmse'])




