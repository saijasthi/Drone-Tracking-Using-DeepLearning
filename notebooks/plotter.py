import math
import matplotlib.pyplot as plt
import numpy as np

class PlotData:
    def __init__(self, X, target, prediction, title=""):
        self.X = X.to_numpy();
        self.target = target.to_numpy();
        self.prediction = prediction;
        self.title = title;

class Plotter:
    def __init__(self):
        pass

    def plot_predictions(self, data, sort=True):
        assert(type(data) is list);
        fig, axes = plt.subplots(math.ceil(len(data) / 3), 3, figsize=(20, 20))
        # fig.figure(figsize=(20, 20))
        for plotData, axis in zip(data, axes.flatten()):
            self.plot_prediction(plotData, axis, sort)
        plt.show()

    def plot_prediction(self, plotData, axis=plt, sort=True):
        x_ax = range(len(plotData.X))
        target_index = np.argsort(plotData.target, axis=0)
        target = plotData.target;
        prediction = plotData.prediction;

        if sort:
            target = plotData.target[target_index]
            prediction = plotData.prediction[target_index]

        axis.scatter(x_ax, target,  s=6, label="target")
        axis.scatter(x_ax, prediction, s=6, label="pred",c="orange", alpha=0.5)
        if hasattr(axis, 'legend'):
            axis.legend()
        if hasattr(axis, 'show'):
            axis.show()

        axis.set_title(plotData.title)
