import random


class MyNeuron:
    def __init__(self, inputs, eta):
        self.w = [random.random()*2]
        self.x = [0]
        self.theta = random.random()*3
        self.eta = eta
        self.inputs = inputs
        for x in range(inputs - 1):
            self.w.append(random.random())
            self.x.append(0)

    def propagate(self):
        y = 0
        u = 0
        for x in self.x:
            u += x * self.w[y]
            y += 1
        return u

    def activate(self):
        if self.propagate() >= self.theta:
            return 1
        else:
            return 0
