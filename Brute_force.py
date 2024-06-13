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
    min_distances_class1 = [float('inf'), float('inf')]
    closest_points_to_class1 = [None, None]
    closest_points_to_class2 = [None, None]
    for i in range(class1_data.shape[0]):
        distance, closest_point = calculate_minimum_distance(class1_data[i], class2_data)
        closest_point_as_tuple = tuple(closest_point)
        closest_points_to_class1_as_tuples = [tuple(point) for point in closest_points_to_class1 if point is not None]
        if distance < min_distances_class1[0] or distance < min_distances_class1[1]:
            if closest_point_as_tuple in closest_points_to_class1_as_tuples:
                index = closest_points_to_class1_as_tuples.index(closest_point_as_tuple)
                min_distances_class1[index] = distance
                closest_points_to_class2[index] = class1_data[i]
            elif min_distances_class1[0] < min_distances_class1[1]:
                min_distances_class1[1] = distance
                closest_points_to_class1[1] = closest_point
                closest_points_to_class2[1] = class1_data[i]
            else:
                min_distances_class1[0] = distance
                closest_points_to_class1[0] = closest_point
                closest_points_to_class2[0] = class1_data[i]
    return np.array(closest_points_to_class1[0]), np.array(closest_points_to_class1[1])

def calculate_best_margin(first_class_support_vectors, second_class_support_vectors):
    distance_between_first_couple = np.linalg.norm(first_class_support_vectors[0] - second_class_support_vectors[0])
    distance_between_second_couple = np.linalg.norm(first_class_support_vectors[1] - second_class_support_vectors[1])
    best_margin = min(distance_between_first_couple, distance_between_second_couple)
    return best_margin

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
    class1_support_vectors = identify_support_vectors(class1_data, class2_data)
    class2_support_vectors = identify_support_vectors(class2_data, class1_data)
    best_margin = calculate_best_margin(class1_support_vectors, class2_support_vectors)
    print(best_margin)