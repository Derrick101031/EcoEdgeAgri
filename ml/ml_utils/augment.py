import numpy as np

def augment(X, noise_std=0.01, dropout_rate=0.05):
    Xn = X + np.random.normal(0, noise_std, X.shape)
    Xn = np.clip(Xn, 0.0, 1.0)
    if dropout_rate:
        mask = np.random.rand(*Xn.shape) < dropout_rate
        Xn[mask] = 0.0
    return Xn

