import numpy as np
import math

results_array = np.genfromtxt('my_file.csv', delimiter=',')

n_rows = results_array.shape[0]
n_cols = results_array.shape[1]

window_size = 6

max_value = 0
max_averages = []

for row in range(((n_rows + 1) - window_size)):
    for col in range(((n_cols + 1) - window_size)):
        average = np.average(results_array[row:(row + window_size)][col:(col + window_size)])
        if math.isclose(average, max_value, rel_tol=1e-4):
            max_averages.append(average)
        elif np.average(results_array[row:(row + window_size)][col:(col + window_size)]) > max_value:
            max_value = average
            max_averages = [average]


