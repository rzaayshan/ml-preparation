import numpy as np
import math
import numpy as np


class Neuron:
    # Don't change anything in the `__init__` function.
    def __init__(self, examples):
        np.random.seed(42)
        # Three weights: one for each feature and one more for the bias.
        self.weights = np.random.normal(0, 1, 3 + 1)
        self.examples = examples
        self.train()

    # Don't use regularization.
    # Use mini-batch gradient descent.
    # Use the sigmoid activation function.
    # Use the defaults for the function arguments.
    def train(self, learning_rate=0.01, batch_size=10, epochs=200):
        # Write your code here.
        mini_batch = []
        for i in range(0, len(self.examples), batch_size):
            mini_batch.append(self.examples[i:i + batch_size])

        for ex in self.examples:
            ex['features'].append(1)

        for epoch in range(0, epochs):
            for batch in mini_batch:
                gradients = [0] * len(self.examples[0]['features'])
                for ex in batch:
                    features = ex['features']

                    result1 = sum(np.array(self.weights) * np.array(features))

                    result2 = 1 / (1 + math.exp(-(result1)))

                    error = result2 - ex['label']

                    for i in range(0, len(features)):
                        gradients[i] += features[i] * error

                gradients = np.array(gradients) / len(batch)
                self.weights = self.weights - learning_rate * gradients

    # Return the probability—not the corresponding 0 or 1 label.
    def predict(self, features):
        # Write your code here.
        features.append(1)
        result1 = sum(np.array(self.weights) * np.array(features))
        result2 = 1 / (1 + math.exp(-(result1)))

        return result2
