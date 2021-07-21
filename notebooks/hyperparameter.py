class HyperParameter:
    def __init__(self, dataset, layers_lambda, layer_descriptor, loss, optimizer, epochs, batch_size):
        self.dataset = dataset;
        self.layers_lambda = layers_lambda;
        self.layer_descriptor = layer_descriptor;
        self.loss = loss;
        self.optimizer = optimizer;
        self.epochs = epochs;
        self.batch_size = batch_size;

    def get_dataset(self):
        return self.dataset;

    def get_layers_lambda(self):
        return self.layers_lambda;

    def get_loss(self):
        return self.loss;

    def get_optimizer(self):
        return self.optimizer;

    def get_epochs(self):
        return self.epochs;

    def get_batch_size(self):
        return self.batch_size;

    def __str__(self):
        return "Parameters: filename={}, layers={}, loss={}, optimizer={}, epochs={}, batch_size={}".format(
            self.dataset.filename,
            self.layer_descriptor,
            self.loss,
            self.optimizer,
            self.epochs,
            self.batch_size
        );

