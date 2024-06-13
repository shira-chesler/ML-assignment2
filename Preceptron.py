import numpy as np


def perceptron_algorithm(data, labels):
    weights = np.zeros(data.shape[1])
    changed_weights = True
    iteration_num = 0
    while changed_weights:
        iteration_num += 1
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
    return weights, iteration_num


def parse_file(path_to_file):
    data = []
    labels = []
    with open(path_to_file) as f:
        for line in f:
            line = line.strip().split()
            data.append([float(x) for x in line[:-1]])
            labels.append(int(line[-1]))
    return np.array(data), np.array(labels)


def get_line_from_weights(weights):
    w1, w2 = weights
    slope = -w1 / w2
    y_intercept = 0
    return slope, y_intercept


def get_distance_from_line(slope, y_intercept, point):
    a = slope
    b = -1
    c = y_intercept
    x, y = point
    distance = abs(a * x + b * y + c) / np.sqrt(a ** 2 + b ** 2)
    return distance


def find_min_distance(weights, data_points):
    slope, y_intercept = get_line_from_weights(weights)
    min_distance = float('inf')
    for point in data_points:
        distance = get_distance_from_line(slope, y_intercept, point)
        if distance < min_distance:
            min_distance = distance
    return min_distance


if __name__ == '__main__':
    data, labels = parse_file('two_circle.txt')
    weights, number_mistakes = perceptron_algorithm(data, labels)
    margin = find_min_distance(weights, data)
    print("Final direction vector: ", weights)
    print("Number of mistakes: ", number_mistakes)
    print("Margin of final vector: ", margin)
