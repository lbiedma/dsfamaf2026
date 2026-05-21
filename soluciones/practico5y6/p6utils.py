import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity


# Definición del clasificador KDE personalizado adaptado del libro de Jake VanderPlas
class KDEClassifier(BaseEstimator, ClassifierMixin):
    """Clasificador Bayesiano Generativo basado en KDE"""
    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        # Se crea un modelo KDE independiente para cada clase presente en las etiquetas
        self.models_ = [KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)
                        for c in self.classes_]
        
        # Se entrena cada modelo KDE únicamente con las muestras correspondientes a su clase
        for c, model in zip(self.classes_, self.models_):
            model.fit(X[y == c])
            
        # Se calculan las probabilidades a priori (log_priors) basadas en la frecuencia de clases
        self.log_priors_ = [np.log(X[y == c].shape[0] / X.shape[0])
                            for c in self.classes_]
        return self
        
    def predict(self, X):
        # Se obtienen las log-verosimilitudes de cada modelo para las nuevas muestras
        logprobs = np.array([model.score_samples(X) for model in self.models_]).T
        # Aplicamos el teorema de Bayes en espacio logarítmico: log(prior) + log(likelihood)
        return self.classes_[np.argmax(logprobs + self.log_priors_, axis=1)]
