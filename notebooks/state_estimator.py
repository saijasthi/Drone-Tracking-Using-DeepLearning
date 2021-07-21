from kerasmodel import KerasModel;
from plotter import Plotter, PlotData;
from metrics import Metrics;
import numpy as np;

class StateEstimator:
    def __init__(self, dataset, layersLambda, loss="mse", optimizer="adam", epochs=30, batch_size=12, verbose=0):
        self.plotter = Plotter();
        self.layersLambda = layersLambda;
        self.loss = loss;
        self.optimizer = optimizer;
        self.epochs = epochs;
        self.batch_size = batch_size;
        self.verbose = verbose;
        self.model_x = KerasModel(layersLambda, loss, optimizer, epochs, batch_size, verbose);
        self.model_y = KerasModel(layersLambda, loss, optimizer, epochs, batch_size, verbose);
        self.model_z = KerasModel(layersLambda, loss, optimizer, epochs, batch_size, verbose);
        self.model_phi = KerasModel(layersLambda, loss, optimizer, epochs, batch_size, verbose);
        self.model_theta = KerasModel(layersLambda, loss, optimizer, epochs, batch_size, verbose);
        self.model_psi = KerasModel(layersLambda, loss, optimizer, epochs, batch_size, verbose);
        self.dataset = dataset;

        self.prediction_x = None;
        self.prediction_y = None;
        self.prediction_z = None;
        self.prediction_phi = None;
        self.prediction_theta = None;
        self.prediction_psi = None;

    def setModels(self, model_x, model_y, model_z, model_phi, model_theta, model_psi):
        self.model_x = model_x;
        self.model_y = model_y;
        self.model_z = model_z;
        self.model_phi = model_phi;
        self.model_theta = model_theta;
        self.model_psi = model_psi;

    def fit(self):
        self.fit_model(self.model_x, self.dataset.getXSplit());
        self.fit_model(self.model_y, self.dataset.getYSplit());
        self.fit_model(self.model_z, self.dataset.getZSplit());
        self.fit_model(self.model_phi, self.dataset.getPhiSplit());
        self.fit_model(self.model_theta, self.dataset.getThetaSplit());
        self.fit_model(self.model_psi, self.dataset.getPsiSplit());

    def fit_model(self, model, train_test_split):
        X_train, X_test, t_train, t_test = train_test_split;
        model.fit(X_train, t_train);

    def predict(self):
        self.prediction_x = self.predict_model(self.model_x, self.dataset.getXSplit());
        self.prediction_y = self.predict_model(self.model_y, self.dataset.getYSplit());
        self.prediction_z = self.predict_model(self.model_z, self.dataset.getZSplit());
        self.prediction_phi = self.predict_model(self.model_phi, self.dataset.getPhiSplit());
        self.prediction_theta = self.predict_model(self.model_theta, self.dataset.getThetaSplit());
        self.prediction_psi = self.predict_model(self.model_psi, self.dataset.getPsiSplit());

    def predict_flight(self, X):
        self.prediction_x = self.predict_model(self.model_x, (None, X, None, None));
        self.prediction_y = self.predict_model(self.model_y, (None, X, None, None));
        self.prediction_z = self.predict_model(self.model_z, (None, X, None, None));
        self.prediction_phi = self.predict_model(self.model_phi, (None, X, None, None));
        self.prediction_theta = self.predict_model(self.model_theta, (None, X, None, None));
        self.prediction_psi = self.predict_model(self.model_psi, (None, X, None, None));

    def predict_model(self, model, train_test_split):
        X_train, X_test, t_train, t_test = train_test_split;
        prediction = model.predict(X_test);
        return prediction

    def get_plot_data(self, model, train_test_split, prediction, title):
        X_train, X_test, t_train, t_test = train_test_split;
        return PlotData(X_test, t_test, prediction, title)

    def plot(self):
        Plotter().plot_predictions([
            self.get_plot_data(self.model_x, self.dataset.getXSplit(), self.prediction_x, "X Results"),
            self.get_plot_data(self.model_y, self.dataset.getYSplit(), self.prediction_y, "Y Results"),
            self.get_plot_data(self.model_z, self.dataset.getZSplit(), self.prediction_z, "Z Results"),
            self.get_plot_data(self.model_phi, self.dataset.getPhiSplit(), self.prediction_phi, "Phi Results"),
            self.get_plot_data(self.model_theta, self.dataset.getThetaSplit(), self.prediction_theta, "Theta Results"),
            self.get_plot_data(self.model_psi, self.dataset.getPsiSplit(), self.prediction_psi, "Psi Results"),
        ]);

    def get_metrics(self):
        X_train, X_test, t_train, t_test = self.dataset.getXSplit();
        results_x = Metrics().get_metrics(t_test, self.prediction_x);
        X_train, X_test, t_train, t_test = self.dataset.getYSplit();
        results_y = Metrics().get_metrics(t_test, self.prediction_y);
        X_train, X_test, t_train, t_test = self.dataset.getZSplit();
        results_z = Metrics().get_metrics(t_test, self.prediction_z);
        X_train, X_test, t_train, t_test = self.dataset.getPhiSplit();
        results_phi = Metrics().get_metrics(t_test, self.prediction_phi);
        X_train, X_test, t_train, t_test = self.dataset.getThetaSplit();
        results_theta = Metrics().get_metrics(t_test, self.prediction_theta);
        X_train, X_test, t_train, t_test = self.dataset.getPsiSplit();
        results_psi = Metrics().get_metrics(t_test, self.prediction_psi);

        return [
            results_x,
            results_y,
            results_z,
            results_phi,
            results_theta,
            results_psi
        ]

    def print_metrics(self):
        Metrics().print_metrics([
            "x",
            "y",
            "z",
            "phi",
            "theta",
            "psi"
        ], self.get_metrics())
