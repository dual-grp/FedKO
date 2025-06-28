import numpy as np
import math
import pandas as pd
import sklearn
from numpy import mean
from sklearn.metrics import roc_curve, roc_auc_score

# TranAD
def adjust_predicts_from_tranad(label, score,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):
    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score > threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict
    
def get_threshold_2(labels, scores, print_or_not=True):
    auc = roc_auc_score(labels, scores)
    thresholds_0 = scores.copy()
    thresholds_0.sort()
    #print(len(thresholds_0))
    thresholds = []
    for i in range(len(thresholds_0)):
        if i % 1000 == 0 or i == len(thresholds_0) - 1:
            thresholds.append(thresholds_0[i])

    best_precision = 0
    best_recall = 0
    best_f1 = 0
    best_threshold = math.inf
    best_f1_adjusted = 0
    best_precision_adjusted = 0
    best_recall_adjusted = 0


    for threshold in thresholds:
        y_pred_from_threshold = [1 if scores[i] >= threshold else 0 for i in range(len(scores))]
        y_pred_from_threshold = np.asarray(y_pred_from_threshold)
        precision = sklearn.metrics.precision_score(labels, y_pred_from_threshold)
        recall = sklearn.metrics.recall_score(labels, y_pred_from_threshold)
        f1 = sklearn.metrics.f1_score(labels, y_pred_from_threshold)

        y_pred_adjusted = adjust_predicts_from_tranad(labels, scores, pred=y_pred_from_threshold, threshold=threshold)
        precision_adjusted = sklearn.metrics.precision_score(labels, y_pred_adjusted)
        recall_adjusted = sklearn.metrics.recall_score(labels, y_pred_adjusted)
        f1_adjusted = sklearn.metrics.f1_score(labels, y_pred_adjusted)

        if f1_adjusted > best_f1_adjusted:
            best_precision = precision
            best_recall = recall
            best_f1 = f1
            best_f1_adjusted = f1_adjusted
            best_precision_adjusted = precision_adjusted
            best_recall_adjusted = recall_adjusted
            best_threshold = threshold


    if print_or_not:
        print('auc:', auc)
        print('precision_adjusted:', best_precision_adjusted)
        print('recall_adjusted:', best_recall_adjusted)
        print('f1:', best_f1)
        print('f1_adjusted:', best_f1_adjusted)
        print('threshold:',  best_threshold)

    return auc, best_precision, best_recall, best_f1, best_precision_adjusted, best_recall_adjusted, best_f1_adjusted