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


def l2norm(point1, point2):
    """
    Calculates the L2 norm between two points.

    Parameters:
    point1 (numpy.ndarray): The first point.
    point2 (numpy.ndarray): The second point.

    Returns:
    float: The L2 norm between the two points.
    """
    return np.linalg.norm(point1 - point2)


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
    min_distance1 = float('inf')
    min_distance2 = float('inf')
    support_vector1_class1 = None
    support_vector1_class2 = None
    support_vector2_class1 = None
    support_vector2_class2 = None

    for i in range(class1_data.shape[0]):
        distance, closest_point = calculate_minimum_distance(class1_data[i], class2_data)
        if distance < min_distance1:
            min_distance2 = min_distance1
            min_distance1 = distance
            support_vector2_class1 = support_vector1_class1
            support_vector2_class2 = support_vector1_class2
            support_vector1_class1 = class1_data[i]
            support_vector1_class2 = closest_point
        elif distance < min_distance2:
            if l2norm(closest_point, class1_data[i]) <= l2norm(closest_point, support_vector1_class1):
                min_distance2 = distance
                support_vector2_class1 = class1_data[i]
                support_vector2_class2 = closest_point

    return support_vector1_class1, support_vector1_class2, support_vector2_class1, support_vector2_class2


def get_points_define_line(point1_class1, point1_class2, point2_class1, point2_class2):
    """
    Calculates the points that define the decision boundary line.

    Parameters:
    point1_class1 (tuple): The first support vector from class 1.
    point1_class2 (tuple): The first support vector from class 2.
    point2_class1 (tuple): The second support vector from class 1.
    point2_class2 (tuple): The second support vector from class 2.

    Returns:
    tuple: The coordinates of the first point that defines the line.
    tuple: The coordinates of the second point that defines the line.
    """
    return ((point1_class1[0] + point1_class2[0]) / 2, (point1_class1[1] + point1_class2[1]) / 2), (
    (point2_class1[0] + point2_class2[0]) / 2, (point2_class1[1] + point2_class2[1]) / 2)


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


def find_min_distance(slope, y_intercept, data_points):
    """
    Finds the minimum distance from any data point to the decision boundary line.

    Parameters:
    slope (float): The slope of the decision boundary.
    y_intercept (float): The y-intercept of the decision boundary.
    data_points (numpy.ndarray): The data points.

    Returns:
    float: The minimum distance from any data point to the line.
    """
    min_distance = float('inf')
    for point in data_points:
        distance = get_distance_from_line(slope, y_intercept, point)
        if distance < min_distance:
            min_distance = distance
    return min_distance


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


'''
def find_highest_and_closest_points(data, labels):
    """
    Finds the highest point from the data points with label -1, and the closest point from the data points with label 1.

    Parameters:
    data (numpy.ndarray): The input data points, where each row represents a data point.
    labels (numpy.ndarray): The corresponding labels for the data points.

    Returns:
    numpy.ndarray: The highest point from the data points with label -1.
    numpy.ndarray: The closest point from the data points with label 1 to the highest point.
    float: The distance between the highest point and the closest point.
    """
    # Separate the data points by their labels
    class1_data = data[labels == 1]
    class_minus1_data = data[labels == -1]

    # Find the highest point from the data points with label -1
    highest_point = class_minus1_data[np.argmax(class_minus1_data[:, 1])]

    # Find the closest point from the data points with label 1
    distances = np.linalg.norm(class1_data - highest_point, axis=1)
    closest_point = class1_data[np.argmin(distances)]

    # Calculate the distance between the highest point and the closest point
    distance = np.linalg.norm(highest_point - closest_point)

    return highest_point, closest_point, distance
'''


def calculate_line_params(point1, point2):
    """
    Calculates the slope and y-intercept of the line passing through two points.

    Parameters:
    point1 (tuple): The coordinates of the first point.
    point2 (tuple): The coordinates of the second point.

    Returns:
    float: The slope of the line.
    float: The y-intercept of the line.
    """
    slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
    y_intercept = point1[1] - slope * point1[0]
    return slope, y_intercept


if __name__ == '__main__':
    # Parse the data from the file
    data_points, data_labels = parse_data_from_file('two_circle.txt')

    # Separate data points by their labels
    class1_data, class2_data = separate_data_by_labels(data_points, data_labels)

    # Identify the support vectors for each class
    support_vector1_class1, support_vector1_class2, support_vector2_class1, support_vector2_class2 = identify_support_vectors(
        class1_data, class2_data)

    # Calculate the line passing through the support vectors
    point1, point2 = get_points_define_line(support_vector1_class1, support_vector1_class2, support_vector2_class1,
                                            support_vector2_class2)

    # Calculate the slope and y-intercept of the line
    slope, y_intercept = calculate_line_params(point1, point2)

    # Find the minimum distance from any data point to the decision boundary line
    best_margin = find_min_distance(slope, y_intercept, data_points)

    print("Optimal margin using brute force: ", best_margin)

    # Find the highest point from the data points with label -1, and the closest point from the data points with label 1
    # highest_point, closest_point, distance = find_highest_and_closest_points(data_points, data_labels)
    # print("Highest point from the data points with label -1: ", highest_point)
    # print("Closest point from the data points with label 1: ", closest_point)
    # print("Distance between the highest point and the closest point divided by 2: ", distance / 2)
