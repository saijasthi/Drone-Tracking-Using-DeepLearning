import pandas as pd
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self, filename, test_size=0.2, random_state=0):
        self.filename = filename;
        self.df = pd.read_csv(filename);
        self.validate_dataframe();
        self.test_size = test_size;
        self.random_state = random_state
        self.X = self.df.iloc[:, 7:]
        self.target = self.df.iloc[:, 1:7]
        
        self.reset();

    def validate_dataframe(self):
        assert(self.df.columns[1] == 'x_change')
        assert(self.df.columns[2] == 'y_change')
        assert(self.df.columns[3] == 'z_change')
        assert(self.df.columns[4] == 'phi_change')
        assert(self.df.columns[5] == 'theta_change')
        assert(self.df.columns[6] == 'psi_change')
        assert((self.df.dtypes != 'float64').sum() == 0) # all columns are floats

    def reset(self):
        self.X_train, self.X_test, self.target_train, self.target_test = train_test_split(self.X, self.target, test_size=self.test_size, random_state=self.random_state)

        self.resetXSplit();
        self.resetYSplit();
        self.resetZSplit();
        self.resetPhiSplit();
        self.resetThetaSplit();
        self.resetPsiSplit();

    def resetXSplit(self):
        self.X_x_train, self.X_x_test, self.target_x_train, self.target_x_test = self.X_train, self.X_test, self.target_train.iloc[:, 0], self.target_test.iloc[:, 0];

    def resetYSplit(self):
        self.X_y_train, self.X_y_test, self.target_y_train, self.target_y_test = self.X_train, self.X_test, self.target_train.iloc[:, 1], self.target_test.iloc[:, 1];

    def resetZSplit(self):
        self.X_z_train, self.X_z_test, self.target_z_train, self.target_z_test = self.X_train, self.X_test, self.target_train.iloc[:, 2], self.target_test.iloc[:, 2];

    def resetPhiSplit(self):
        self.X_phi_train, self.X_phi_test, self.target_phi_train, self.target_phi_test = self.X_train, self.X_test, self.target_train.iloc[:, 3], self.target_test.iloc[:, 3];

    def resetThetaSplit(self):
        self.X_theta_train, self.X_theta_test, self.target_theta_train, self.target_theta_test = self.X_train, self.X_test, self.target_train.iloc[:, 4], self.target_test.iloc[:, 4];

    def resetPsiSplit(self):
        self.X_psi_train, self.X_psi_test, self.target_psi_train, self.target_psi_test = self.X_train, self.X_test, self.target_train.iloc[:, 5], self.target_test.iloc[:, 5];

    def getXSplit(self):
        return self.X_x_train, self.X_x_test, self.target_x_train, self.target_x_test;

    def getYSplit(self):
        return self.X_y_train, self.X_y_test, self.target_y_train, self.target_y_test;

    def getZSplit(self):
        return self.X_z_train, self.X_z_test, self.target_z_train, self.target_z_test;

    def getPhiSplit(self):
        return self.X_phi_train, self.X_phi_test, self.target_phi_train, self.target_phi_test;

    def getThetaSplit(self):
        return self.X_theta_train, self.X_theta_test, self.target_theta_train, self.target_theta_test;

    def getPsiSplit(self):
        return self.X_psi_train, self.X_psi_test, self.target_psi_train, self.target_psi_test;

    def getInDimension(self):
        return self.X.shape[1]

    def getOutDimension(self):
        return 1