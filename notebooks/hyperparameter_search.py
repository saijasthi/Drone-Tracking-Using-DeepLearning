import time, sys
from IPython.display import clear_output
from state_estimator import StateEstimator;
from hyperparameter import HyperParameter;
import time;

MAX_FAILURES = 6

class HyperParameterSearch:
    def __init__(self):
        self.results = {};
        self.start_time = time.perf_counter();

    def createParameterSet(self, dataset_lambda_pair_set, loss_set, optimizer_func_set, epoch_set, batch_size_set):
        parameters = []
        for dataset_lambda_pair in dataset_lambda_pair_set:
            (dataset, lambda_func, name) = dataset_lambda_pair
            for loss_val in loss_set:
                for optimizer_func in optimizer_func_set:
                    for epoch_val in epoch_set:
                        for batch_size_val in batch_size_set:
                            parameters.append(
                                HyperParameter(
                                    dataset,
                                    lambda_func,
                                    name,
                                    loss=loss_val, 
                                    optimizer=optimizer_func(),
                                    epochs=epoch_val,
                                    batch_size=batch_size_val
                                )
                            )
        return parameters

    def run(self, parameters):
        self.best_x_score = float("inf")
        self.best_x_name = "";
        self.best_y_score = float("inf")
        self.best_y_name = "";
        self.best_z_score = float("inf")
        self.best_z_name = "";
        self.best_phi_score = float("inf")
        self.best_phi_name = "";
        self.best_theta_score = float("inf")
        self.best_theta_name = "";
        self.best_psi_score = float("inf")
        self.best_psi_name = "";
        self.start_time = time.perf_counter();
        parameter_index = 0
        failure_count = 0;
        

        while(parameter_index < len(parameters)):
            self.update_progress(parameter_index / len(parameters));
            if (failure_count >= MAX_FAILURES):
                print("Parameter reach {} failures: {}".format(failure_count, str(parameters[parameter_index])))
                failure_count = 0
                parameter_index += 1

            try:
                if (parameter_index >= len(parameters)):
                    break;

                parameter = parameters[parameter_index];
                estimator = StateEstimator(
                    parameter.get_dataset(),
                    parameter.get_layers_lambda(),
                    parameter.get_loss(),
                    parameter.get_optimizer(),
                    parameter.get_epochs(),
                    parameter.get_batch_size()
                )
                estimator.fit();
                estimator.predict();
                metrics = estimator.get_metrics();
                
                if (metrics[0][5] < self.best_x_score):
                    self.best_x_score = metrics[0][5];
                    self.best_x_name = str(parameter);
                if (metrics[1][5] < self.best_y_score):
                    self.best_y_score = metrics[1][5];
                    self.best_y_name = str(parameter);
                if (metrics[2][5] < self.best_z_score):
                    self.best_z_score = metrics[2][5];
                    self.best_z_name = str(parameter);
                if (metrics[3][5] < self.best_phi_score):
                    self.best_phi_score = metrics[3][5];
                    self.best_phi_name = str(parameter);
                if (metrics[4][5] < self.best_theta_score):
                    self.best_theta_score = metrics[4][5];
                    self.best_theta_name = str(parameter);
                if (metrics[5][5] < self.best_psi_score):
                    self.best_psi_score = metrics[5][5];
                    self.best_psi_name = str(parameter);

                parameter_index += 1;
                failure_count = 0
                self.results[str(parameter)] = {
                    'estimator': estimator,
                    'metrics': metrics,
                };
            except ValueError as err:
                failure_count += 1
                parameter.get_dataset().reset();
                print("Parameter failed, ", parameter, "Error", err)

        self.update_progress(parameter_index / len(parameters));

        return self.results;

    def update_progress(self, progress):
        bar_length = 20
        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            progress = 0
        if progress < 0:
            progress = 0
        if progress >= 1:
            progress = 1

        block = int(round(bar_length * progress))

        clear_output(wait = True)
        text = "Progress: [{0}] {1:.1f}%, elapsed: {2:.4f}".format( "#" * block + "-" * (bar_length - block), progress * 100, time.perf_counter() - self.start_time)
        print(text)

    def print(self):
        print("X:     {}, {}\nY:     {}, {}\nZ:     {}, {}\nPhi:   {}, {}\nTheta: {}, {}\nPsi:   {}, {}\n".format(
            self.best_x_score,
            self.best_x_name,
            self.best_y_score,
            self.best_y_name,
            self.best_z_score,
            self.best_z_name,
            self.best_phi_score,
            self.best_phi_name,
            self.best_theta_score,
            self.best_theta_name,
            self.best_psi_score,
            self.best_psi_name
        ));