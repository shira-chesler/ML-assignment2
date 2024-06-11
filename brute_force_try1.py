import numpy as np


def separate_classes(data, labels):
    class1 = []
    class2 = []
    for i in range(len(labels)):
        if labels[i] == 1:
            class1.append(data[i])
        else:
            class2.append(data[i])
    return np.array(class1), np.array(class2)


def find_distance(param, data_other_class):
    min_distance = float('inf')
    closest_point = None
    for i in range(data_other_class.shape[0]):
        distance = np.linalg.norm(param - data_other_class[i])
        if distance < min_distance:
            min_distance = distance
            closest_point = data_other_class[i]
    return min_distance, closest_point


def find_support_vectors(data_class1, data_class2):
    class1_min_distances = [float('inf'), float('inf')]
    closest_points_to_class1 = [None, None]
    closest_points_to_class2 = [None, None]
    for i in range(data_class1.shape[0]):
        distance, closest_point = find_distance(data_class1[i], data_class2)
        if distance < class1_min_distances[0] or distance < class1_min_distances[1]:
            if closest_point in closest_points_to_class1:
                if closest_point == closest_points_to_class1[0]:
                    class1_min_distances[0] = distance
                    closest_points_to_class2[0] = data_class1[i]
                else:
                    class1_min_distances[1] = distance
                    closest_points_to_class2[1] = data_class1[i]
            elif class1_min_distances[0] < class1_min_distances[1]:
                class1_min_distances[1] = distance
                closest_points_to_class1[1] = closest_point
                closest_points_to_class2[1] = data_class1[i]
            else:
                class1_min_distances[0] = distance
                closest_points_to_class1[0] = closest_point
                closest_points_to_class2[0] = data_class1[i]
    return np.array(closest_points_to_class1[0]), np.array(closest_points_to_class1[1])


def find_center_and_radius_of_circle(point1, point2, point3):
    center_x, center_y, r = 0, 0, 0
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3
    A = x2 - x1
    B = y2 - y1
    C = x3 - x1
    D = y3 - y1
    E = A * (x1 + x2) + B * (y1 + y2)
    F = C * (x1 + x3) + D * (y1 + y3)
    G = 2 * (A * (y3 - y2) - B * (x3 - x2))
    if G == 0:
        return center_x, center_y, r
    center_x = (D * E - B * F) / G
    center_y = (A * F - C * E) / G
    r = np.sqrt((center_x - x1) ** 2 + (center_y - y1) ** 2)
    return center_x, center_y, r


def find_mid_point_on_circle(two_points, center, radius):
    x1, y1 = two_points[0]
    x2, y2 = two_points[1]
    cx, cy = center

    theta1 = np.arctan2(y1 - cy, x1 - cx)
    theta2 = np.arctan2(y2 - cy, x2 - cx)

    theta_mid = (theta1 + theta2) / 2

    mid_x = cx + radius * np.cos(theta_mid)
    mid_y = cy + radius * np.sin(theta_mid)

    return mid_x, mid_y


def find_best_margin(first_class_support_vectors, second_class_support_vectors, center, radius):
    first_first_dist = np.sqrt(np.power(first_class_support_vectors[0][0] - second_class_support_vectors[0][0], 2) +
                               np.power(first_class_support_vectors[0][1] - second_class_support_vectors[0][1], 2))
    first_second_dist = np.sqrt(
        np.power(first_class_support_vectors[1][0] - second_class_support_vectors[1][0], 2) + np.power(
            first_class_support_vectors[1][1] - second_class_support_vectors[1][1], 2))

    if first_first_dist < first_second_dist:
        first_couple = [first_class_support_vectors[0], second_class_support_vectors[0]]
        second_couple = [first_class_support_vectors[1], second_class_support_vectors[1]]
    else:
        first_couple = [first_class_support_vectors[1], second_class_support_vectors[0]]
        second_couple = [first_class_support_vectors[1], second_class_support_vectors[0]]

    distance1 = find_mid_point_on_circle(first_couple, center, radius)
    distance2 = find_mid_point_on_circle(second_couple, center, radius)

    best_margin = min(distance1, distance2)
    return best_margin

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
    class1, class2 = separate_classes(data, labels)
    support_vectors_class1 = find_support_vectors(class1, class2)
    support_vectors_class2 = find_support_vectors(class2, class1)
    center, radius = find_center_and_radius_of_circle(support_vectors_class1[0], support_vectors_class1[1],
                                                      support_vectors_class2[0])
    best_margin = find_best_margin(support_vectors_class1, support_vectors_class2, center, radius)
    print(best_margin)