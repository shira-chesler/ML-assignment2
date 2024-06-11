import numpy as np


def create_line(point1, point2):
    slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
    y_intercept = point1[1] - slope * point1[0]
    return slope, y_intercept


def find_relevant_lines(data, labels):
    line_set = []
    for i in range(data.shape[0]):
        for j in range(i, data.shape[0]):
            if i == j:
                continue
            line = create_line(data[i], data[j])
            if (is_seperating(line, data, labels)):
                line_set.append(line)
    return line_set
