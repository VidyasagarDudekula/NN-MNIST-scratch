import numpy as np


def preprocess_line(line):
    if line is None or isinstance(line, str) is False:
        raise ValueError("invalid line value")
    line_values = [int(i.strip()) for i in line.strip().split(',')]
    label = line_values[0]
    pixel_values = line_values[1:]
    return (pixel_values, label)

def preprocess_input(pixel_values):
    if pixel_values is None:
        return None
    if isinstance(pixel_values, list) is False:
        raise ValueError("Input pixel array should be a list")
    new_array = np.array(pixel_values, dtype=float)
    new_array = (new_array/255.0 * 0.99) + 0.01
    return new_array.tolist()

def preprocess_output(value, onodes=10):
    if value is None or isinstance(value, int) is False or value < 0:
        raise ValueError("Invalue value")
    
    target = np.zeros(onodes, dtype=float) + 0.01
    target[value] = 0.99
    return target.tolist()

    