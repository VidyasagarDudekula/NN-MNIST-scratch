import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit, logit
import json
from data_preprocessing import preprocess_input, preprocess_line, preprocess_output


class NeuralNetworks:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.lr = learningrate

        self.activation_function = lambda x: expit(x)
        self.inverse_activation_function = lambda x: logit(x)

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
    

    def backquery(self, target_list):
        if len(target_list)!=self.onodes:
            raise ValueError("target list length didn't match with output nodes")
        
        final_outputs = np.array(target_list, ndmin=2).T
        final_inputs = self.inverse_activation_function(final_outputs)

        hidden_outputs = np.dot(self.who.T, final_inputs)

        hidden_outputs -= np.min(hidden_outputs)
        hidden_outputs /= np.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01

        hidden_inputs = self.inverse_activation_function(hidden_outputs)
        inputs = np.dot(self.wih.T, hidden_inputs)
        inputs -= np.min(inputs)
        inputs /= np.max(inputs)
        inputs *= 0.98
        inputs += 0.01
        
        return inputs
    
    
def test_network(nn_obj):
    if nn_obj is None:
        return None
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
        predicted_list = nn_obj.query(input_list)
        index = int(np.argmax(predicted_list).astype(int))
        data['target'] = line_data[1]
        data['predicted'] = index
        data['predicted_list'] = predicted_list.tolist()
        data['result'] = data['target'] == data['predicted']
        if data['result']:
            correct += 1
        test_results.append(data)
    accuracy = correct/len(test_data_lines)
    return accuracy


def generate_stats(stats):
    plt.figure(figsize=(10, 6))

    x_hidden = [d['hidden_count'] for d in stats]
    y_best = [d['best_accuracy'] for d in stats]

    plt.bar(x_hidden, y_best, width=60, color='skyblue', edgecolor='black', alpha=0.8)
    plt.xlabel('Hidden Node Count', fontsize=12)
    plt.ylabel('Best Accuracy', fontsize=12)
    plt.title('Best Accuracy vs Hidden Node Count', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for i, v in enumerate(y_best):
        plt.text(x_hidden[i], v, f"{v:.4f}", ha='center', va='bottom', fontsize=9)

    plt.savefig('hidden_vs_best_accuracy.png')
    plt.close()
    print("Saved hidden_vs_best_accuracy.png")

    plt.figure(figsize=(12, 7))

    for entry in stats:
        hidden_count = entry['hidden_count']
        epoch_stats = entry['epoch_level_stats']
        
        x_epoch = [e['epoch'] for e in epoch_stats]
        y_accuracy = [e['accuracy'] for e in epoch_stats]
        
        plt.plot(x_epoch, y_accuracy, marker='o', label=f'Hidden: {hidden_count}')

    plt.xlabel('Epoch Number', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy Progression by Epoch', fontsize=14)
    plt.legend(title="Hidden Nodes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.savefig('epoch_vs_accuracy_combined.png')
    plt.close()
    print("Saved epoch_vs_accuracy_combined.png")


def train_network(inputnodes, hiddennodes, outputnodes, learningrate, epochs=4):
    nn = NeuralNetworks(inputnodes=inputnodes, hiddennodes=hiddennodes, outputnodes=outputnodes, learningrate=learningrate)
    data_file = open('mnist_train.csv', 'r')
    data_lines = data_file.readlines()[1:]
    data_file.close()

    for _ in range(epochs):
        for line in data_lines:
            line_data = preprocess_line(line)
            input_list = preprocess_input(line_data[0])
            target_list = preprocess_output(line_data[1])
            nn.train(input_list, target_list)
    return nn


def test_with_different_hidden_counts():
    inputnodes = 784
    outputnodes = 10
    hiddennodes_list = [i for i in range(100, 501, 100)]
    learningrate = 0.012

    stats = []

    for hiddennodes in hiddennodes_list:
        print(f"For hidennodes count:- {hiddennodes}")
        nn = NeuralNetworks(inputnodes=inputnodes, hiddennodes=hiddennodes, outputnodes=outputnodes, learningrate=learningrate)
        current_stats = {'hidden_count': hiddennodes}
        data_file = open('mnist_train.csv', 'r')
        data_lines = data_file.readlines()[1:]
        data_file.close()

        epoch = 6
        best_acc = 0
        total_epoch_stats = []
        for ep in range(epoch):
            epoch_stats = {'epoch': ep+1}
            for line in data_lines:
                line_data = preprocess_line(line)
                input_list = preprocess_input(line_data[0])
                target_list = preprocess_output(line_data[1])
                nn.train(input_list, target_list)
            accuracy = test_network(nn) # return is floating value (correct/total)
            epoch_stats['accuracy'] = accuracy
            best_acc = max(best_acc, accuracy)
            total_epoch_stats.append(epoch_stats)
        current_stats["epoch_level_stats"] = total_epoch_stats
        current_stats['best_accuracy'] = best_acc
        stats.append(current_stats)
    return stats
    
                
def generate_input(target_value):
    inputnodes = 784
    hiddennodes = 200
    outputnodes = 10
    epochs = 5
    learningrate = 0.012
    nn_onj = train_network(inputnodes=inputnodes, hiddennodes=hiddennodes, outputnodes=outputnodes, epochs=epochs, learningrate=learningrate)
    target_list = [0.01 for i in range(10)]
    target_list[target_value] = 0.99
    input_matrix = nn_onj.backquery(target_list)

    plt.imshow(input_matrix.reshape(28,28), cmap='gray', interpolation='None')
    plt.savefig('gen_input_image.png')




if __name__ == '__main__':
    # inputnodes = 784
    # hiddennodes = 100
    # outputnodes = 10
    # learningrate = 0.012
    # nn = NeuralNetworks(inputnodes=inputnodes, hiddennodes=hiddennodes, outputnodes=outputnodes, learningrate=learningrate)

    # data_file = open('mnist_train.csv', 'r')
    # data_lines = data_file.readlines()[1:]
    # data_file.close()

    # epoch = 2
    # for ep in range(epoch):
    #     for line in data_lines:
    #         line_data = preprocess_line(line)
    #         input_list = preprocess_input(line_data[0])
    #         target_list = preprocess_output(line_data[1])
    #         nn.train(input_list, target_list)
    
    # # test

    # test_data_file = open('mnist_test.csv', 'r')
    # test_data_lines = test_data_file.readlines()[1:]
    # test_data_file.close()

    # test_results = []
    # correct = 0
    # for index, line in enumerate(test_data_lines):
    #     data = {}
    #     data['index'] = index
    #     line_data = preprocess_line(line)
    #     input_list = preprocess_input(line_data[0])
    #     predicted_list = nn.query(input_list)
    #     index = int(np.argmax(predicted_list).astype(int))
    #     data['target'] = line_data[1]
    #     data['predicted'] = index
    #     data['predicted_list'] = predicted_list.tolist()
    #     data['result'] = data['target'] == data['predicted']
    #     if data['result']:
    #         correct += 1
    #     test_results.append(data)
    # total_data = {'predictions': test_results, 'accuracy': correct/len(test_data_lines)}
    # with open('test_all_output_200_hidden_nodes.json', 'w') as json_file:
    #     json.dump(total_data, json_file, indent=4)

    # stats = test_with_different_hidden_counts()
    # with open('stats.json', 'w') as json_file:
    #     json.dump(stats, json_file, indent=4)
    # generate_stats(stats)

    generate_input(8)
    


        
