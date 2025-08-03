import pandas as pd
import numpy as np

def load_noisy_dataset(path):
    df = pd.read_csv(path)
    features = df.drop(columns=['label']).values
    labels = df['label'].values
    noisy_labels = labels.copy()
    noise = np.random.rand(len(labels)) < 0.2
    noisy_labels[noise] = 1 - noisy_labels[noise]
    return features, noisy_labels
