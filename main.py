import numpy as np


class NeuronLayer:
    def __init__(self, weights, biases) -> None:
        self.__weights = weights
        self.__biases = biases
        self.__shape = weights.shape

    @staticmethod
    def __sigmoid(x):
        return np.exp(-2.5 * (x) ** 2) / np.sqrt(0.4)

    @staticmethod
    def __sigmoid_derivative(x):
        return NeuronLayer.__sigmoid(x) * -5 * (x)

    def get_shape(self):
        return self.__shape

    def __getitem__(self, item):
        if item[0] == 'w':
            return self.__weights[item[1] // self.__shape[1], item[1] % self.__shape[1]]
        elif item[0] == 'b':
            return self.__biases[item[1]]
        raise ValueError('Invalid type of item')

    def __setitem__(self, item, value):
        if item[0] == 'w':
            self.__weights[item[1] // self.__shape[1], item[1] % self.__shape[1]] = value
        elif item[0] == 'b':
            self.__biases[item[1]] = value
        else:
            raise ValueError('Invalid type of item')

    def __call__(self, inputs, is_deriv=False):
        if is_deriv:
            return self.__sigmoid_derivative(np.dot(self.__weights, inputs) + self.__biases)
        else:
            return self.__sigmoid(np.dot(self.__weights, inputs) + self.__biases)


class NeuronNetwork:
    def __init__(self, layers):
        self.__layers = layers
        self.amount_of_layers = len(layers)

    def __call__(self, inputs, for_layer=-1, is_last_deriv=False):
        if for_layer == -1:
            for_layer = self.amount_of_layers
        if for_layer == 0:
            return self.__layers[0](inputs, is_deriv=is_last_deriv)
        for layer in self.__layers[:for_layer - int(is_last_deriv)]:
            inputs = layer(inputs)
        if is_last_deriv:
            return self.__layers[for_layer - 1](inputs, is_deriv=True)
        return inputs

    def __getitem__(self, item):
        if item < self.amount_of_layers:
            return self.__layers[item]
        raise ValueError('Invalid layer index')

    def __setitem__(self, item, value):
        if item < self.amount_of_layers:
            self.__layers[item] = value
        else:
            raise ValueError('Invalid layer index')

    @staticmethod
    def __sigmoid_derivative(x):
        return np.exp(-2.5 * (x) ** 2) * -5 * (x) / np.sqrt(0.4)

    @staticmethod
    def __mse(true, pred):
        return np.mean((true - pred) ** 2)

    @staticmethod
    def __mult(x):
        res = 1
        for i in x:
            res *= i
        return res

    def recursive_part_of_part_dev(self, output, needed_item, inputs):
        output_layer, output_num_neuron = output[0], output[1]
        type_of_item, item_layer, item_num_neuron = needed_item[0], needed_item[1], needed_item[2]
        current_layer = self.__layers[output_layer]
        needed_weights = current_layer._NeuronLayer__weights
        needed_shape = needed_weights.shape

        if item_layer > output_layer:
            return 0

        if item_layer == output_layer:
            match type_of_item:
                case 'w':
                    if not item_layer:
                        return self(inputs, for_layer=output_layer, is_last_deriv=True)[item_num_neuron // needed_shape[0], 0] *\
                            inputs[item_num_neuron // needed_shape[1], 0]
                    return self(inputs, for_layer=output_layer, is_last_deriv=True)[item_num_neuron // needed_shape[0], 0] *\
                        self(inputs, for_layer=output_layer - 1)[item_num_neuron // needed_shape[1], 0]
                case 'b':
                    return self(inputs, for_layer=output_layer, is_last_deriv=True)[item_num_neuron // needed_shape[0], 0]
                case _:
                    raise ValueError('Choose right type of value (w or b)')


        return self(inputs, for_layer=output_layer, is_last_deriv=True)[output_num_neuron, 0] *\
            np.dot(np.array([self.recursive_part_of_part_dev((output_layer - 1, i), needed_item, inputs) for i in range(needed_shape[0])]), needed_weights[:, output_num_neuron].reshape(needed_shape[0], 1))[0]

    def get_partial_dev(self, true, output, needed_item, inputs):
        return (-2 * (true - self(inputs)[output[1]]) * self.recursive_part_of_part_dev(output, needed_item, inputs))[0]

    def train_values(self, true_datas, output, all_inputs, limit=1000, eta=0.1, is_print=False):
        for epoch in range(limit):
            for layer_num in range(len(self.__layers)):
                shape = self.__layers[layer_num].get_shape()
                all_adresses = [('w', layer_num, i) for i in range(shape[0] * shape[1])] + [('b', layer_num, i) for i in range(shape[0])]
                for value_adress in all_adresses:
                    adress_for_layer = (value_adress[0], value_adress[2])
                    self.__layers[layer_num][adress_for_layer] = self.__layers[layer_num][adress_for_layer] - eta * sum(np.array([self.get_partial_dev(true, output, value_adress, pred) for true, pred in zip(true_datas, all_inputs)]))

            if is_print:
                print(f"epoch: {epoch + 1}\n"
                      f"MSE: {self.__mse(np.array(true_datas), np.array(list(map(self, all_inputs))))}\n"
                      f"{'-' * 30}")


def main():
    neurons_layer3 = NeuronLayer(np.random.rand(1, 2), np.random.rand(1))
    neuron_network = NeuronNetwork([NeuronLayer(np.random.rand(2, 2), np.random.rand(2, 1)) for i in range(1)] + [neurons_layer3])
    all_inputs = (np.array([[0], [0]]), np.array([[0], [1]]), np.array([[1], [0]]), np.array([[1], [1]]))
    true_data = (0, 0, 0, 1)
    neuron_network.train_values(true_data, (1, 0), all_inputs, is_print=True, limit=2000, eta=0.5)

    print('Try inputs\n')
    while True:
        print(neuron_network(np.array([[float(i)] for i in input().split()])))


if __name__ == "__main__":
    main()
