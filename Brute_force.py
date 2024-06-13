import numpy as np


def separate_data_by_labels(data, labels):
    """
    Separates the data points into two classes based on their labels.

    Parameters:
    data (numpy.ndarray): The input data points, where each row represents a data point.
    labels (numpy.ndarray): The corresponding labels for the data points.

    Returns:
    numpy.ndarray: Data points belonging to class 1.
    numpy.ndarray: Data points belonging to class 2.
    """
    class1_data = []
    class2_data = []
    for i in range(len(labels)):
        if labels[i] == 1:
            class1_data.append(data[i])
        else:
            class2_data.append(data[i])
    return np.array(class1_data), np.array(class2_data)


def calculate_minimum_distance(point, other_class_data):
    """
    Calculates the minimum distance from a given point to a set of points from another class.

    Parameters:
    point (numpy.ndarray): The point from one class.
    other_class_data (numpy.ndarray): The data points from the other class.

    Returns:
    float: The minimum distance to the closest point in the other class.
    numpy.ndarray: The closest point in the other class.
    """
    minimum_distance = float('inf')
    closest_point = None
    for i in range(other_class_data.shape[0]):
        distance = np.linalg.norm(point - other_class_data[i])
        if distance < minimum_distance:
            minimum_distance = distance
            closest_point = other_class_data[i]
    return minimum_distance, closest_point


def identify_support_vectors(class1_data, class2_data):
    """
    Identifies the support vectors, which are the closest points between two classes.

    Parameters:
    class1_data (numpy.ndarray): Data points of the first class.
    class2_data (numpy.ndarray): Data points of the second class.

    Returns:
    numpy.ndarray: The support vector from the first class.
    numpy.ndarray: The support vector from the second class.
    """
    min_distance = float('inf')
    support_vector_class1 = None
    support_vector_class2 = None

    for i in range(class1_data.shape[0]):
        distance, closest_point = calculate_minimum_distance(class1_data[i], class2_data)
        if distance < min_distance:
            min_distance = distance
            support_vector_class1 = class1_data[i]
            support_vector_class2 = closest_point

    return support_vector_class1, support_vector_class2


def calculate_best_margin(class1_support_vectors, class2_support_vectors):
    """
    Calculates the optimal margin between the closest support vectors of two classes.

    Parameters:
    class1_support_vectors (numpy.ndarray): The support vector from the first class.
    class2_support_vectors (numpy.ndarray): The support vector from the second class.

    Returns:
    float: The distance between the two support vectors, representing the margin.
    """
    distance_between_vectors = np.linalg.norm(class1_support_vectors - class2_support_vectors)
    return distance_between_vectors


def parse_data_from_file(file_path):
    """
    Parses a file containing data points and labels.

    Parameters:
    file_path (str): The path to the file containing the data.

    Returns:
    numpy.ndarray: The parsed data points.
    numpy.ndarray: The parsed labels.
    """
    data_points = []
    data_labels = []
    with open(file_path) as file:
        for line in file:
            line = line.strip().split()
            data_points.append([float(x) for x in line[:-1]])
            data_labels.append(int(line[-1]))
    return np.array(data_points), np.array(data_labels)


if __name__ == '__main__':
    # Parse the data from the file
    data_points, data_labels = parse_data_from_file('two_circle.txt')

    # Separate data points by their labels
    class1_data, class2_data = separate_data_by_labels(data_points, data_labels)

    # Identify the support vectors for each class
    class1_support_vector, class2_support_vector = identify_support_vectors(class1_data, class2_data)

    # Calculate the optimal margin between the support vectors
    best_margin = calculate_best_margin(class1_support_vector, class2_support_vector)

    print("Optimal margin using brute force: ", best_margin)
