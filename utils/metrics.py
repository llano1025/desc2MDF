from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
import os


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


def find_latest_file(file_path, model_type):
    file_list = os.listdir(file_path)
    matching_files = [file for file in file_list if
                      file.startswith(f'{model_type}_finetuned_BERT_epoch_') and file.endswith('.pth')]

    if matching_files:
        numbers = [int(Path(file).stem.split("_")[-1]) for file in matching_files]
        max_number = max(numbers)
        file_with_max_number = f'{model_type}_finetuned_BERT_epoch_{max_number}.pth'
        print(f"The file with the largest number is: {file_with_max_number}")
        return file_with_max_number, max_number
    else:
        print("No matching files found in the directory.")
        return None, None

def parse_to_result(parsed_list, MDF2Rec_api_prediction, Desc2MDF_api_prediction):
    merged_array = []
    for desc, group_no, recommended_form in zip(parsed_list, Desc2MDF_api_prediction, MDF2Rec_api_prediction):
        merged_array.append({
            'desc': desc,
            'groupNo': group_no,
            'recommendedForm': recommended_form
        })
    return merged_array
