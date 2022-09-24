from improvement import train_and_predict
from read_data import read_dataset, split_dataset
from numpy.random import default_rng
import numpy as np
from evaluate import compute_accuracy

if __name__ == "__main__":
    print("Loading the training dataset...")

    (x, y, classes) = read_dataset("data/train_full.txt")

    (x_test, y_test, classes_test) = read_dataset("data/test.txt")

    (x_hyp_val, y_hyp_val, classes_val) = read_dataset("data/validation.txt")

    # Generate a validation set
    # 0.20 reserved for validation. must take 0.4125 of remaining
    # test set to train tree on 33% bootstrap data
    seed = 60025
    rg = default_rng(seed)
    x_train, x_train_val, y_train, y_train_val = split_dataset(x, y, 0.2, rg)

    # hyperparameter list to iterate over
    hyp_list = {
        "total_trees": [x for x in range(50, 50, 10)],
        "p_value": [x for x in range(4, 4, 10)],
        "bootstrap_proportion": list(np.arange(0.3, 0.3, 0.05)),
        "pruning_proportion": list(np.arange(0.5, 1.00, 0.05)),
    }

    # storing the results of each hyperparameter run
    results = {
        "total_trees": [],
        "p_value": [],
        "bootstrap_proportion": [],
        "pruning_proportion": [],
    }

    for hyp_name, hyp_values in hyp_list.items():
        print("Hyperparameter being tested: ", hyp_name)
        for hyp_value in hyp_values:

            # run train_and predict with that hyperparameter
            predictions = train_and_predict(
                x_train, y_train, x_test, x_train_val, y_train_val, hyp_value, hyp_name
            )
            # calculate accuracy with returned prediction
            results[hyp_name].append(compute_accuracy(y_test, predictions))

    # check the output
    print("Hyperparameter tuning results: ")
    print(hyp_list)
    print(results)