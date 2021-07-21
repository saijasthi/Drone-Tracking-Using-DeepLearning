from tensorflow.keras.models import Sequential

class KerasModel:
    """ 
    Model class to simplify the creation and handling process 

    layers: a list of all of the layers e.g.
        [
            Dense(256, input_dim=in_dim, activation="relu"),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dense(out_dim)
        ]
    """

    def __init__(self, layersLambda, loss="mse", optimizer="adam", epochs=30, batch_size=12, verbose=0):
        self.loss = loss
        self.optimizer = optimizer
        self.layersLambda = layersLambda;
        self.epochs=epochs
        self.batch_size=batch_size
        self.verbose=verbose
        self.reinitialize();

    def reinitialize(self):
        self.model = Sequential()
        layers = self.layersLambda()
        for layer in layers:
            self.model.add(layer)
            
        self.model.compile(loss=self.loss, optimizer=self.optimizer)

    def fit(self, X, t):
        self.model.fit(X, t, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)

    def predict(self, X):
        return self.model.predict(X)
