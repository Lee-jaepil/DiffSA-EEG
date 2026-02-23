import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix

def load_evaluation_dataset(file_path):
    data = np.load(file_path)
    return torch.from_numpy(data['data']).float(), torch.from_numpy(data['labels']).long(), data['file_paths']

def calculate_metrics(y_true, y_pred, y_pred_proba, num_classes=2):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)

    try:
        if num_classes == 2:
            # Binary classification
            auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        else:
            # Multi-class classification
            y_true_onehot = np.eye(num_classes)[y_true]
            auc = roc_auc_score(y_true_onehot, y_pred_proba, average='macro', multi_class='ovo')
    except ValueError as e:
        print(f"Error calculating AUC: {e}")
        auc = float('nan')

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if num_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
    else:
        # Macro-average specificity for multi-class
        specificities = []
        for i in range(num_classes):
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            specificity_i = tn / (tn + fp) if (tn + fp) != 0 else 0
            specificities.append(specificity_i)
        specificity = np.mean(specificities)

    return {
        'accuracy': accuracy,
        'f1': f1,
        'recall': recall,
        'precision': precision,
        'auc': auc,
        'specificity': specificity
    }
