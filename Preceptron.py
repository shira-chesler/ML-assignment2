import numpy as np


def perceptron_algorithm(data, labels):
    """
    Implements the Perceptron algorithm to find the optimal weights for binary classification.

    Parameters:
    data (numpy.ndarray): The input data points, where each row represents a data point.
    labels (numpy.ndarray): The corresponding labels for the data points, assumed to be -1 or 1.

    Returns:
    numpy.ndarray: The final weights after training.
    int: The number of iterations until convergence.
    """
    weights = np.zeros(data.shape[1])  # Initialize weights as a zero vector
    changed_weights = True  # Flag to check if weights have changed in an iteration
    error_num = 0  # Counter for the number of iterations
    while changed_weights:
        changed_weights = False
        for i in range(data.shape[0]):
            prediction = np.dot(data[i], weights)
            if prediction > 0:
                prediction = 1
            else:
                prediction = -1
            if prediction != labels[i]:
                error_num += 1
                if labels[i] == 1:
                    weights += data[i]  # Update weights for positive label
                else:
                    weights -= data[i]  # Update weights for negative label
                changed_weights = True  # Indicate that weights have changed
                break  # Break to start the next iteration
    return weights, error_num


def parse_file(path_to_file):
    """
    Parses a file containing data points and labels.

    Parameters:
    path_to_file (str): The path to the file containing the data.

    Returns:
    numpy.ndarray: The parsed data points.
    numpy.ndarray: The parsed labels.
    """
    data = []
    labels = []
    with open(path_to_file) as f:
        for line in f:
            line = line.strip().split()
            data.append([float(x) for x in line[:-1]])
            labels.append(int(line[-1]))
    return np.array(data), np.array(labels)


def get_line_from_weights(weights):
    """
    Calculates the slope and y-intercept of the decision boundary line from the weights.

    Parameters:
    weights (numpy.ndarray): The weights of the decision boundary.

    Returns:
    float: The slope of the line.
    float: The y-intercept of the line.
    """
    w1, w2 = weights
    slope = -w1 / w2
    y_intercept = 0
    return slope, y_intercept


def get_distance_from_line(slope, y_intercept, point):
    """
    Calculates the perpendicular distance from a point to a line.

    Parameters:
    slope (float): The slope of the line.
    y_intercept (float): The y-intercept of the line.
    point (tuple): The coordinates of the point.

    Returns:
    float: The perpendicular distance from the point to the line.
    """
    a = slope
    b = -1
    c = y_intercept
    x, y = point
    distance = abs(a * x + b * y + c) / np.sqrt(a ** 2 + b ** 2)
    return distance


def find_min_distance(weights, data_points):
    """
    Finds the minimum distance from any data point to the decision boundary line.

    Parameters:
    weights (numpy.ndarray): The weights of the decision boundary.
    data_points (numpy.ndarray): The data points.

    Returns:
    float: The minimum distance from any data point to the line.
    """
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
