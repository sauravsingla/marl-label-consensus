import pandas as pd
from sklearn.datasets import make_classification

def generate_noisy_dataset(path='data/external/noisy_dataset.csv', n_samples=200, noise_ratio=0.2):
    X, y = make_classification(n_samples=n_samples, n_features=5, n_classes=2, random_state=42)
    df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(X.shape[1])])
    df['label'] = y
    noise_indices = df.sample(frac=noise_ratio, random_state=1).index
    df.loc[noise_indices, 'label'] = 1 - df.loc[noise_indices, 'label']
    df.to_csv(path, index=False)
