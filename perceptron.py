import numpy as np

# Normalize a vector to length 1
def unit(v):
    return v / np.sqrt(np.dot(v, v))

def train(data, targets, learning_rate=0.01, threshold=0.5):
    vec_width = data[0].shape
    weights = np.zeros(vec_width)

    while True:
        # Assume solved until we find otherwise
        solved = True

        for (value, target) in zip(data, targets):
            # Grab the unit vector of features
            value = np.asarray(value)
            if not np.any(value == 0):
                value = unit(value)

            error = target - perceive(weights, value, threshold)

            if error != 0:
                solved = False
                weights += learning_rate * error * value

        if solved:
            break

    return weights

def perceive(weights, vec, threshold=0.5):
    return int(np.dot(weights, unit(vec)) > threshold)
