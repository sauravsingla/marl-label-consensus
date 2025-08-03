from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(true_labels, predicted_labels):
    return {
        'accuracy': accuracy_score(true_labels, predicted_labels),
        'precision': precision_score(true_labels, predicted_labels),
        'recall': recall_score(true_labels, predicted_labels),
        'f1_score': f1_score(true_labels, predicted_labels)
    }
