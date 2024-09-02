from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

def test(model, X_test, y_test, verbose = False):
    final_preds = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, pred_proba[:,1])
    scores = []
    scores.append(accuracy_score(y_test, final_preds))
    scores.append(recall_score(y_test, final_preds))
    scores.append(precision_score(y_test, final_preds))
    scores.append(f1_score(y_test, final_preds))
    scores.append(matthews_corrcoef(y_test, final_preds))
    scores.append(roc_auc_score(y_test, pred_proba[:,1]))
    if verbose:
        print(f"Accuracy: {scores[0]}")
        print(f"Confusion matrix: \n{confusion_matrix(y_test, final_preds)}")
        print(f"Recall: {scores[1]}")
        print(f"Precision: {scores[2]}")
        print(f"f1: {scores[3]}")
        print(f"MCC: {scores[4]}")
        print(f"AUC: {scores[5]}")
    return np.array(scores), [fpr, tpr, thresholds]