import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
import json
from data_preprocessing import preprocess_input, preprocess_line, preprocess_output


class NeuralNetworks:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.lr = learningrate

        self.activation_function = lambda x: expit(x)

        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

    def train(self, input_lists, target_lists):
        inputs = np.array(input_lists, ndmin=2).T
        targets = np.array(target_lists, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs

        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0-final_outputs)), np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0-hidden_outputs)), np.transpose(inputs))



    def query(self, input_list):
        inputs = np.array(input_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


if __name__ == '__main__':
    inputnodes = 784
    hiddennodes = 100
    outputnodes = 10
    learningrate = 0.012
    nn = NeuralNetworks(inputnodes=inputnodes, hiddennodes=hiddennodes, outputnodes=outputnodes, learningrate=learningrate)

    data_file = open('mnist_train.csv', 'r')
    data_lines = data_file.readlines()[1:]
    data_file.close()

    epoch = 2
    for ep in range(epoch):
        for line in data_lines:
            line_data = preprocess_line(line)
            input_list = preprocess_input(line_data[0])
            target_list = preprocess_output(line_data[1])
            nn.train(input_list, target_list)
    
    # test

    test_data_file = open('mnist_test.csv', 'r')
    test_data_lines = test_data_file.readlines()[1:]
    test_data_file.close()

    test_results = []
    correct = 0
    for index, line in enumerate(test_data_lines):
        data = {}
        data['index'] = index
        line_data = preprocess_line(line)
        input_list = preprocess_input(line_data[0])
        predicted_list = nn.query(input_list)
        index = int(np.argmax(predicted_list).astype(int))
        data['target'] = line_data[1]
        data['predicted'] = index
        data['predicted_list'] = predicted_list.tolist()
        data['result'] = data['target'] == data['predicted']
        if data['result']:
            correct += 1
        test_results.append(data)
    total_data = {'predictions': test_results, 'accuracy': correct/len(test_data_lines)}
    with open('test_all_output.json', 'w') as json_file:
        json.dump(total_data, json_file, indent=4)
    


        
