import numpy as np

def majority_vote(predictions):
    # predictions: shape (num_agents, num_samples)
    return np.round(np.mean(predictions, axis=0)).astype(int)
