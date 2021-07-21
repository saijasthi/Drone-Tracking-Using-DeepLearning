from sklearn.metrics import r2_score, max_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error;
import numpy as np

class Metrics:
    def __init__(self):
        self.eval_names = ["R2", "max_err", "MAE", "MAPE", "MSE", "RMSE"]

    def get_metrics(self, target, prediction):
        return [
            r2_score(target, prediction),
            max_error(target, prediction),
            mean_absolute_error(target, prediction),
            mean_absolute_percentage_error(target, prediction),
            mean_squared_error(target, prediction),
            np.square(mean_squared_error(target, prediction)),
        ]

    def print_metrics(self, names, metrics):
        print("\t\t{}\t\t{}\t\t{}\t\t{}\t\t{}\t\t{}".format(*self.eval_names))
        for name, results in zip(names, metrics):
            print(name, end="\t\t")
            print("{:.2e}\t{:.2e}\t{:.2e}\t{:.2e}\t{:.2e}\t{:.2e}".format(*results))
