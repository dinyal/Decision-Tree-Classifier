##############################################################################
# Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the train_and_predict() function.
#             You are free to add any other methods as needed.
##############################################################################

import numpy as np
import collections
from evaluate import compute_accuracy
import math
from random_forest import RandomForestClassifier


def train_and_predict(
    x_train, y_train, x_test, x_val, y_val, hyp_value=None, hyp_name=None
):

    assert x_train.shape[0] == len(y_train) and x_val.shape[0] == len(
        y_val
    ), "Training failed. x and y must have the same number of instances."

    # testing hyper-parameter inputs
    hyperparameters = {
        "p_value": 7,
        "total_trees": 170,
        "bootstrap_proportion": 0.6,
        "pruning_proportion": 0.1,
    }

    # this will only run if you're passing in hyperparams, else we use defaults
    if hyp_value and hyp_name:
        hyperparameters[hyp_name] = hyp_value

    # Initialise new random forest classifier class
    random_forest = RandomForestClassifier(
        hyperparameters["total_trees"],
        hyperparameters["p_value"],
        hyperparameters["bootstrap_proportion"],
        hyperparameters["pruning_proportion"],
    )

    # run classifier: Random forest classifier object stores every tree generated in a list
    random_forest.run_forest(x_train, y_train)

    # new list of predictions
    predictions_list = []

    # run every model in the classifier's tree list, prune first, and then add prediction
    # predictions list
    for i, model in enumerate(random_forest.models):
        # We use our validation set that is completely distinct from the data we used to
        # train the models in our random forest. We prune based on model performance on
        # this validation set

        random_forest.prune_nodes(x_val, y_val, model)

        predictions_test = random_forest.improved_predict(x_test, model)
        predictions_list.append(predictions_test)

    # Store the most commonly occuring prediction into new prediction array
    avg_predictions = np.zeros((x_test.shape[0],), dtype=np.object)

    # count and return most commonly occuring label
    for i in range(len(predictions_list[0])):
        cnt = collections.Counter()
        for j in range(len(predictions_list)):
            cnt[predictions_list[j][i]] += 1
        avg_predictions[i] = cnt.most_common(1)[0][0]

    return avg_predictions
