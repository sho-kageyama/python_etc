from neuron import neuron


class NeuralNetwork:
    neuron = neuron()
    w_im = [[0.496, 0.512], [-0.501, 0.998],
            [0.498, -0.502]]
    w_mo = [0.121, -0.4996, 0.200]

    input_layer = [0.0, 0.0, 1.0]
    middle_layer = [[neuron, neuron, 1.0] * 3]
    output_layer = neuron

    def setNeuron(self, num, input_data):
        self.input_layer[0] = input_data[0]
        self.input_layer[1] = input_data[1]

        for middle in self.middle_layer:
            for layer in middle:
                if layer == 1.0:
                    pass
                else:
                    layer.reset()

        self.output_layer.reset()

    def commit(self, input_data):

        self.setNeuron()

        for middle in self.middle_layer:
            middle[0].setInput(self.input_layer[0] * self.w_im[0][0])
            middle[0].setInput(self.input_layer[1] * self.w_im[1][0])
            middle[0].setInput(self.input_layer[2] * self.w_im[2][0])

            middle[1].setInput(self.input_layer[0] * self.w_im[0][1])
            middle[1].setInput(self.input_layer[1] * self.w_im[1][1])
            middle[1].setInput(self.input_layer[2] * self.w_im[2][1])

        for middle in self.middle_layer:
            self.output_layer.setInput(middle[0].getOutput() * self.w_mo[0])
            self.output_layer.setInput(middle[1].getOutput() * self.w_mo[1])
            self.output_layer.setInput(middle[2] * self.w_mo[2])

        return self.output_layer.getOutput()

    def learn(self, input_data):
        output_data = self.commit([input_data[0], input_data[1]])

        correct_value = input_data[2]

        k = 0.3

        delta_w_mo = (correct_value - output_data) * output_data * (1.0 - output_data)
        old_w_mo = list(self.w_mo)
        for middle in self.middle_layer:
            self.w_mo[0] += middle[0].output * delta_w_mo * k
            self.w_mo[0] += middle[1].output * delta_w_mo * k
            self.w_mo[0] += middle[2] * delta_w_mo * k

        delta_w_1 = 0.0
        delta_w_2 = 0.0
        for middle in self.middle_layer:
            delta_w_1 += delta_w_mo * old_w_mo[0] * middle[0].output * (1.0 - middle[0].output)
            delta_w_2 += delta_w_mo * old_w_mo[1] * middle[1].output * (1.0 - middle[1].output)

        delta_w_im = [
            delta_w_1,
            delta_w_2,
        ]
        self.w_im[0][0] += self.input_layer[0] * delta_w_im[0] * k
        self.w_im[0][1] += self.input_layer[0] * delta_w_im[1] * k
        self.w_im[1][0] += self.input_layer[1] * delta_w_im[0] * k
        self.w_im[1][1] += self.input_layer[1] * delta_w_im[1] * k
        self.w_im[2][0] += self.input_layer[2] * delta_w_im[0] * k
        self.w_im[2][1] += self.input_layer[2] * delta_w_im[1] * k
