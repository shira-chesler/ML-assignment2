import numpy as np


def perceptron_algorithm(data, labels):
    weights = np.zeros(data.shape[1])
    changed_weights = True
    while changed_weights:
        changed_weights = False
        for i in range(data.shape[0]):
            prediction = np.dot(data[i], weights)
            if prediction > 0:
                prediction = 1
            else:
                prediction = -1
            if prediction != labels[i]:
                if labels[i] == 1:
                    weights += data[i]
                    changed_weights = True
                else:
                    weights -= data[i]
                    changed_weights = True
                break
    return weights


def parse_file(path_to_file):
    data = []
    labels = []
    with open(path_to_file) as f:
        for line in f:
            line = line.strip().split()
            data.append([float(x) for x in line[:-1]])
            labels.append(int(line[-1]))
    return np.array(data), np.array(labels)


if __name__ == '__main__':
    data, labels = parse_file('two_circle.txt')
    weights = perceptron_algorithm(data, labels)
    print(weights)
