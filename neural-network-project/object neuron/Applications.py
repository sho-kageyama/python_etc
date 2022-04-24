import random
import math
import matplotlib.pyplot as plt
from Neural_Network import NeuralNetwork


class Applications:

    refer_point0 = 34.5
    refer_point1 = 137.5

    position_tokyo_learned = [[], []]
    position_kanagawa_learned = [[], []]

    position_tokyo_learning = [[],[]]
    position_kanagawa_learning = [[],[]]

    training_data = []

    neural_network = NeuralNetwork()


    def sigmoid(a):
        return 1.0 / (1.0 + math.exp(-a))


    training_data_file = open("training_data","r")
    for line in training_data_file:
        line = line.rstrip().strip(",")
        training_data.append([float(line[0])-refer_point0, float(line[1])-refer_point1, int(line[2])])
    training_data_file.close()



    def learning(self):
        for i in range(0, 1000):
            for data in self.traing_data:
                NeuralNetwork.learn(data)

        print(NeuralNetwork.w_im)
        print(NeuralNetwork.w_mo)



    def commitment(self, num):
        data_to_commit = list()
        for i in range(0,num):
            data1 = random.uniform(34.0,36.0)
            data2 = random.uniform(137.0,139.0)
            data_to_commit.append([data1,data2])

        for data in data_to_commit:
            data[0] -= self.refer_point0
            data[1] -= self.refer_point1


        for data in data_to_commit:
            if NeuralNetwork.commit(data) < 0.5:
                self.position_tokyo_learned[0].append(data[1] + self.refer_point1)
                self.position_tokyo_learned[1].append(data[0] + self.refer_point0)
            else:
                self.position_kanagawa_learned[0].append(data[1] + self.refer_point1)
                self.position_kanagawa_learned[1].append(data[0] + self.refer_point0)


        for data in self.training_data:
            if data[2] < 0.5:
                self.position_tokyo_learning[0].append(data[1] + self.refer_point1)
                self.position_tokyo_learning[1].append(data[0] + self.refer_point0)
            else:
                self.position_kanagawa_learning[0].append(data[1] + self.refer_point1)
                self.position_kanagawa_learning[0].append(data[1] + self.refer_point1)





    def prot(self):
        plt.scatter(self.position_tokyo_learning[0], self.position_tokyo_learning[1], c="red", label="Tokyo_learn", maker="+")
        plt.scatter(self.position_kanagawa_learning[0], self.position_kanagawa_learning[1], c="blue", label="Kanagawa-learn", maker="+")
        plt.scatter(self.position_tokyo_learned[0], self.position_tokyo_learned[1], c="red", label="Tokyo", maker="o")
        plt.scatter(self.position_kanagawa_learned[0], self.position_kanagawa_learned[1], c="blue", label="Kanagawa", maker="o")

        plt.legend()
        plt.show()