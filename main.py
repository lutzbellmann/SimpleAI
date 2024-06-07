# Artificial Intelligence for solving logic problems
# Emulates a single perceptron that is configurable and trainable
# to solve 2-dimensional separable problems (like AND, OR, NAND,...)
# visualization of decision lines before and after learning
# by Dr. Lutz Bellmann

import matplotlib.pyplot as plt
import numpy as np
import neuron


def learn(perceptron, omegal, target):
    print("Input   Ergebnis   Soll")
    ok = True
    for t in range(len(target)):
        for i in range(perceptron.inputs):
            perceptron.x[i] = omegal[t][i]
            print(str(omegal[t][i]) + "   ", end="")
        y = perceptron.activate()
        print("    " + str(y) + "        " + str(target[t]), end="")
        print()
        delta = target[t] - y
        if delta != 0:
            ok = False
        perceptron.theta -= perceptron.eta * delta
        for i in range(perceptron.inputs):
            perceptron.w[i] += perceptron.eta * delta * omegal[t][i]
    print()
    return ok


def printstats(perceptron):
    j = 1
    for i in perceptron.w:
        print("w" + str(j) + ": " + str(round(i, 2)))
        j += 1
    print("Theta: " + str(round(perceptron.theta, 2)))
    print("eta: " + str(perceptron.eta))
    print()


def basediagram(target):
    plt.title("Neuron Plot", size=28)
    plt.xlabel("Input 1 (x1)", size=20)
    plt.ylabel("Input 2 (x2)", size=20)
    plt.axis([-1, 3, -1, 3])
    plt.grid(color='green', linestyle='--', linewidth=0.5)
    x = np.array([0, 0, 1, 1])
    y = np.array([0, 1, 0, 1])
    colors = np.empty(len(target), dtype=str)
    for i in range(len(target)):
        if target[i] == 1:
            colors[i] = "red"
        else:
            colors[i] = "blue"
    plt.scatter(x, y, c=colors, s=100)


def plotline(perceptron, color, stat):
    x = np.linspace(-1, 3, 50, endpoint=True)
    y = -(perceptron.w[0] / perceptron.w[1]) * x + (perceptron.theta / perceptron.w[1])
    plt.plot(x, y, lw="3", c=color, label=stat)


# input the Perceptron parameters here:
# plotting the output will only work for two inputs


inputs = 2                                     # no of inputs
eta = 0.0937                                   # learning rate
omegal = [[0, 0], [0, 1], [1, 0], [1, 1]]      # input training vector
target = [0, 1, 1, 1]                          # expected result vector

MyPerceptron = neuron.MyNeuron(inputs, eta)
basediagram(target)
plotline(MyPerceptron, "blue", "before learning")
printstats(MyPerceptron)
i = 2
print("1. Durchgang")
while not (learn(MyPerceptron, omegal, target) or i > 100):
    print(str(i) + ". Durchgang:")
    i += 1

printstats(MyPerceptron)
plotline(MyPerceptron, "red", "after learning")
plt.legend(loc="upper left")
plt.show()
