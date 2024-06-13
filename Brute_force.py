import numpy as np


def separate_data_by_labels(data, labels):
    class1_data = []
    class2_data = []
    for i in range(len(labels)):
        if labels[i] == 1:
            class1_data.append(data[i])
        else:
            class2_data.append(data[i])
    return np.array(class1_data), np.array(class2_data)


def calculate_minimum_distance(point, other_class_data):
    minimum_distance = float('inf')
    closest_point = None
    for i in range(other_class_data.shape[0]):
        distance = np.linalg.norm(point - other_class_data[i])
        if distance < minimum_distance:
            minimum_distance = distance
            closest_point = other_class_data[i]
    return minimum_distance, closest_point


def identify_support_vectors(class1_data, class2_data):
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
    distance_between_vectors = np.linalg.norm(class1_support_vectors - class2_support_vectors)
    return distance_between_vectors


def parse_data_from_file(file_path):
    data_points = []
    data_labels = []
    with open(file_path) as file:
        for line in file:
            line = line.strip().split()
            data_points.append([float(x) for x in line[:-1]])
            data_labels.append(int(line[-1]))
    return np.array(data_points), np.array(data_labels)


if __name__ == '__main__':
    data_points, data_labels = parse_data_from_file('two_circle.txt')
    class1_data, class2_data = separate_data_by_labels(data_points, data_labels)
    class1_support_vector, class2_support_vector = identify_support_vectors(class1_data, class2_data)
    best_margin = calculate_best_margin(class1_support_vector, class2_support_vector)
    print("Optimal margin using brute force: ", best_margin)
