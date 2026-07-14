import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    y_pred = np.array(y_pred)

    # Prevent log(0)
    y_pred = np.clip(y_pred, 1e-15, 1)

    # Probability of the true class for each sample
    probs = y_pred[np.arange(len(y_true)), y_true]

    # Average negative log likelihood
    return -np.mean(np.log(probs))