import numpy as np
X = np.array(([0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]), dtype=float)
y = np.array(([1], [0], [0], [0], [0], [0], [0], [1]), dtype=float)

xPredicted = np.array(([0, 0, 1]), dtype=float)

X = X/np.amax(X, axis=0)
xPredicted = xPredicted/np.amax(xPredicted, axis=0)

class Neural_Network (object):
    def __init__(self):
        self.inputLayerSize = 3
        self.outputLayerSize = 1
        self.hiddenLayerSize = 4
    
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

        self.lossFile = open("SumSquaredLossList.csv", "a")

        
    def feedForward(self, X):
        self.z = np.dot(X, self.W1)
        self.z2 = self.activationSigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2)
        o = self.activationSigmoid(self.z3)
        return o

    def backwardPropagate(self, X, y, o):
        self.o_error = y - o
        self.o_delta = self.o_error * self.activationSigmoidPrime(o)
        self.z2_error = self.o_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error * self.activationSigmoidPrime(self.z2)
        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.o_delta)
    
    def trainNetwork(self, X, y):
        o = self.feedForward(X)
        self.backwardPropagate(X, y, o)

    def activationSigmoid(self, s):
        return 1/(1+np.exp(-s))
    
    def activationSigmoidPrime(self, s):
        return s * (1 - s)
    
    def saveSumSquaredLossList(self, i, error):
        self.lossFile.write(str(i) + ", " + str(error.tolist())+"\n")
        self.lossFile.flush()

    def saveWeights(self):
        np.savetxt("weightsLayer1.txt", self.W1, fmt = "%s")
        np.savetxt("weightsLayer2.txt", self.W2, fmt = "%s")
    
    def predictOutput(self):
        print("Predicted XOR output data based on trained weights: ")
        print("Expected (X1 - X3): \n" + str(xPredicted))
        print("Output (Y1): \n" + str(self.feedForward(xPredicted)))

    def closeFile(self):
        self.lossFile.close()

myNeuralNetwork = Neural_Network()
trainingEpochs = 1000000
for i in range(trainingEpochs):
    print("Epoch # " + str(i) + "\n")
    print("Network Input: \n" + str(X))
    print("Expected Output of XOR Gate Neural Network: \n" + str(y))
    print("Actual Output from XOR Gate Neural Network: \n" + str(myNeuralNetwork.feedForward(X)))

    Loss = np.mean(np.square(y - myNeuralNetwork.feedForward(X)))
    myNeuralNetwork.saveSumSquaredLossList(i, Loss)
    print("Sum Squared Loss: \n" + str(Loss))
    print("\n")
    myNeuralNetwork.trainNetwork(X, y)

myNeuralNetwork.saveWeights()
myNeuralNetwork.predictOutput()
myNeuralNetwork.closeFile()