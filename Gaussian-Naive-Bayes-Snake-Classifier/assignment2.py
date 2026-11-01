"""COMP 3106 Assignment 2: Gaussian Naive Bayes snake classifier.

Classify a snake from its length, weight, and speed using parameters
estimated from a labelled CSV training dataset.
"""

import csv
import math
import statistics

SNAKE_CLASSES = ("anaconda", "cobra", "python")


def naive_bayes_classifier(dataset_filepath, snake_measurements):
    """Return (predicted_class, [P(anaconda), P(cobra), P(python)]).

    The CSV has no header and contains class, length, weight, and speed
    on each row. The sample standard deviation is used for each feature.
    """
    # Collect the three measurements for every snake of each species.
    training_data = {snake_class: [] for snake_class in SNAKE_CLASSES}
    with open(dataset_filepath, newline="", encoding="utf-8-sig") as csv_file:
        for row in csv.reader(csv_file):
            if not row:
                continue
            snake_class = row[0].strip()
            measurements = [float(value) for value in row[1:4]]
            training_data[snake_class].append(measurements)

    total_snakes = sum(len(snakes) for snakes in training_data.values())
    log_scores = []

    for snake_class in SNAKE_CLASSES:
        class_data = training_data[snake_class]

        # Prior: P(class) = number of snakes in class / total snakes.
        prior = len(class_data) / total_snakes
        if not class_data:
            log_scores.append(float("-inf"))
            continue
        log_score = math.log(prior)

        # Under Naive Bayes, the three feature likelihoods are multiplied.
        # Taking logs converts their product to a sum and avoids underflow.
        for feature_index, measurement in enumerate(snake_measurements):
            feature_values = [snake[feature_index] for snake in class_data]
            mean = statistics.mean(feature_values)
            standard_deviation = statistics.stdev(feature_values)

            # Prevent division by zero if every training value is identical.
            if standard_deviation == 0:
                standard_deviation = 1e-9

            z_score = (measurement - mean) / standard_deviation
            log_density = (
                -math.log(standard_deviation * math.sqrt(2 * math.pi))
                - 0.5 * z_score * z_score
            )
            log_score += log_density

        log_scores.append(log_score)

    # Normalize with a numerically stable softmax to get probabilities.
    max_log_score = max(log_scores)
    scaled_scores = [math.exp(score - max_log_score) for score in log_scores]
    score_total = sum(scaled_scores)
    class_probabilities = [score / score_total for score in scaled_scores]

    best_class_index = class_probabilities.index(max(class_probabilities))
    most_likely_class = SNAKE_CLASSES[best_class_index]
    return most_likely_class, class_probabilities
