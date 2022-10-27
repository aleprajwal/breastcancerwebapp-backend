from flask_restful import Resource
from flask import request, jsonify
import pickle
import numpy as np


class BreastCancerCassifier(Resource):
    def __init__(self):
        filename = './classifier/fineTunedModel.pkl'
        self.network = pickle.load(open(filename, 'rb'))

    def get(self):
        retMsg = {"Message": "[Info] perform post request with breastcancer tumor features value"}
        return jsonify(retMsg)

    def post(self):
        # get posted data by user
        print(request.data)
        input_features = request.get_json(force=True)
        data = []
        for _, value in input_features.items():
            data.append(int(value))
        normalized_data = self.normalizeDataset(data)
        output = self.forward_propagate(self.network, normalized_data)
        
        if output[0] >= 0.7:
            message = 'Malignant'
        elif output[0] <= 0.3:
            message = 'Benign'
        else:
            message = 'Inconclusive'
        
        retMsg = {
            'Result': output[0],
            'Message': message,
            'Status': 200
        }
        return jsonify(retMsg)

    def normalizeDataset(self, dataset):
        dataset = np.array(dataset)
        normalize = (dataset - 1) / (10 - 1)
        return normalize

    # Calculate neuron activation for an input
    def activate(self, weights, inputs):
        activation = weights[-1]
        for i in range(len(weights) - 1):
            activation += weights[i] * inputs[i]
        return activation

    # Transfer neuron activation
    def transfer(self, activation):
        return 1.0 / (1.0 + np.exp(-activation))

    # Forward propagate input to a network output
    def forward_propagate(self, network, row):
        inputs = row
        for layer in network:
            new_inputs = []
            for neuron in layer:
                activation = self.activate(neuron['weights'], inputs)
                neuron['output'] = self.transfer(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs

'''
def readJson(filename='./json/data.json'):
    dataset = []
    with open(filename, 'r') as file:
        data = json.load(file)
    for key, value in data.items():
        value = int(value)
        dataset.append(value)
    return dataset


def main():
    dataset = readJson()
    dataset = normalizeDataset(dataset)
    output = forward_propagate(network, dataset)
    result = {'result': output[0]}
    json.dump(result, open('./json/result.json', 'w'))
    print(output)
    if output[0] >= 0.7:
        print('Malignant')
    elif output[0] <= 0.3:
        print('Benign')
    else:
        print('Inconclusive')
'''
