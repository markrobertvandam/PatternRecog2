import numpy as np
import math

results_array = np.genfromtxt('data/results/cats/sift_acc_svm.csv', delimiter=',')

print(np.shape(results_array))
n_rows = results_array.shape[0]
n_cols = results_array.shape[1]

window_size = 1

max_value = 0
max_averages = []
max_region = []

print(results_array[1:2, 1:2])

for row in range(((n_rows + 1) - window_size)):
    for col in range(((n_cols + 1) - window_size)):
        average = np.average(results_array[row:(row + window_size), col:(col + window_size)])
        if math.isclose(average, max_value, rel_tol=1e-3):
            max_averages.append(average)
            max_region.append([row + 1, col + 1])
        elif average > max_value:
            max_value = average
            max_averages = [average]
            max_region = [[row + 1, col + 1]]

print("Max averages regions are ", max_averages)
print("Max region rows and cols are ", max_region)
