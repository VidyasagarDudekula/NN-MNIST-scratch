import numpy as np
import matplotlib.pyplot as plt

data_file = open('mnist_train.csv', 'r')
data_list = data_file.readlines()
data_file.close()

for line in data_list[1:]:
    line_values = [int(i.strip()) for i in line.strip().split(',')]
    label = line_values[0]
    pixel_values = line_values[1:]
    image_array = np.array(pixel_values).reshape((28, 28))
    plt.imshow(image_array, cmap="gray", interpolation='None')
    plt.title(f"Value:- {label}")
    plt.show()
    break
