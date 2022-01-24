import numpy as np
import math

results_array = np.genfromtxt('data/results/cats/fourier_f1_svm.csv', delimiter=',')

# results_array = results_array[:, 78]
# print(results_array.shape)
# print(results_array)

data_dimension = 2

n_rows = results_array.shape[0]

window_size_row = 1

max_value = 0
max_averages = []
max_region = []
if data_dimension == 1:
    for row in range(((n_rows + 1) - window_size_row)):
        average = np.average(results_array[row:(row + window_size_row)])
        if math.isclose(average, max_value, rel_tol=1e-2):
            max_averages.append(average)
            max_region.append([row + 1])
            max_value = max(max_averages)
        elif average > max_value:
            max_value = average
            max_averages = [average]
            max_region = [[row + 1]]

    print("Max averages regions are ", max_averages)
    print("Max region rows and cols are ", max_region)

elif data_dimension == 2:

    n_cols = results_array.shape[1]
    window_size_col = 1

    for row in range(((n_rows + 1) - window_size_row)):
        for col in range(((n_cols + 1) - window_size_col)):
            average = np.average(results_array[row:(row + window_size_row), col:(col + window_size_col)])
            if math.isclose(average, max_value, rel_tol=1e-3):
                max_averages.append(average)
                max_region.append([row + 1, col + 1])
                max_value = max(max_averages)
            elif average > max_value:
                max_value = average
                max_averages = [average]
                max_region = [[row + 1, col + 1]]

    # zipped_lists = zip(max_averages, max_region)
    #
    # sorted_pairs = sorted(zipped_lists)
    #
    # tuples = zip(*sorted_pairs)
    #
    # max_averages, max_region = [list(tup) for tup in tuples]
    #
    # print(max_averages[17])
    # print(max_region[17])

    print("Maxorder = np.max_averages.arg averages regions are ", max_averages)
    print("Max region rows and cols are ", max_region)

elif data_dimension == 3:
    print("max value is ", np.max(results_array))

    kernel_slice = 24
    c_slice = 6

    for kernel in range(4):
        data_kernel = results_array[(kernel * kernel_slice):((kernel * kernel_slice) + kernel_slice)]
        print(data_kernel.shape)
        print("kernel ", kernel)
        for c in range(4):
            data_kernel_slice = data_kernel[(c * c_slice):((c * c_slice) + c_slice)]
            print("slice ", c)
            print("max value is ", np.max(data_kernel_slice))
            print("average of slice ", np.average(data_kernel_slice))

elif data_dimension == 4:
    print("max value is ", np.max(results_array))

    kernel_slice = 24
    c_slice = 6
    gamma_range = 6

    for kernel in range(4):
        data_kernel = results_array[:, (kernel * kernel_slice):((kernel * kernel_slice) + kernel_slice)]
        print(data_kernel.shape)
        print("kernel ", kernel)
        for c in range(4):
            data_kernel_slice = data_kernel[:, (c * c_slice):((c * c_slice) + c_slice)]
            max_value = 0
            max_averages = []
            max_region = []
            print("slice ", c)
            print("max value is ", np.max(data_kernel_slice))
            for row in range(((n_rows + 1) - window_size_row)):
                average = np.average(data_kernel_slice[row:(row + window_size_row), :])
                if math.isclose(average, max_value, rel_tol=1e-3):
                    max_averages.append(average)
                    max_region.append([row + 1])
                    max_value = max(max_averages)
                elif average > max_value:
                    max_value = average
                    max_averages = [average]
                    max_region = [[row + 1]]
            print("Max averages regions are ", max_averages)
            print("Max region rows and cols are ", max_region)
