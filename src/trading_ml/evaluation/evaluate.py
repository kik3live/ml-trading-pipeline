from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def evaluate_model(y_true, y_pred, y_prob):

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }

    return metrics