from Applications import Applications


class neuron:


    input_sum = 0.0
    output = 0.0

    def setInput(self, inp):
        self.input_sum += inp



    def reset(self):
        self.input_sum = 0.0
        self.output = 0.0


    def getOutput(self):
        app = Applications()
        self.output = app.sigmoid(self.input_sum)
        return self.output

