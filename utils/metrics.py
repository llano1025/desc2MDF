import numpy as np
from sklearn.metrics import f1_score


class BertMetrics:
    def __init__(self, preds, labels):
        self.preds = preds
        self.labels = labels
        self.f1_score = self.f1_score_func()

    def f1_score_func(self):
        preds_flat = np.argmax(self.preds, axis=1).flatten()
        labels_flat = self.labels.flatten()
        return f1_score(labels_flat, preds_flat, average='weighted')

    def accuracy_per_class(self, label_dict):
        label_dict_inverse = {v: k for k, v in label_dict.items()}

        preds_flat = np.argmax(self.preds, axis=1).flatten()
        labels_flat = self.labels.flatten()

        for label in np.unique(labels_flat):
            y_preds = preds_flat[labels_flat == label]
            y_true = labels_flat[labels_flat == label]
            print(f'Class: {label_dict_inverse[label]}')
            print(f'Accuracy: {len(y_preds[y_preds == label])}/{len(y_true)}\n')


