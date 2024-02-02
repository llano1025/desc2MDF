import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer


class Dataloader_Desc2MDF:
    def __init__(self, model_name, bert_path):
        self.model_name = model_name
        self.bert_path = bert_path
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name, cache_dir=self.bert_path)
        self.dataset_train, self.dataset_val, self.label_dict, self.size = self.prepare_data()
        self.batch_size = 32
        self.dataloader_train, self.dataloader_validation = self.batch_data()

    def prepare_data(self):
        # Read excel / read dataframe
        df = pd.read_excel(r"data/BasicInfo_Result.xlsx")
        X = df['ASSET_DESCRIPTION'].tolist()

        # Encoding the Labels
        possible_labels = df['GROUPNO'].unique()
        label_dict = {}
        for index, possible_label in enumerate(possible_labels):
            label_dict[possible_label] = index
        y = df['GROUPNO'].replace(label_dict).tolist()

        # Prepare training data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

        # Tokenize and encode training and testing data
        train_encodings = self.tokenizer.batch_encode_plus(X_train,
                                                           add_special_tokens=True,
                                                           return_attention_mask=True,
                                                           pad_to_max_length=True,
                                                           max_length=64,
                                                           return_tensors='pt')

        input_ids_train = train_encodings['input_ids']
        attention_masks_train = train_encodings['attention_mask']
        labels_train = torch.tensor(y_train)

        test_encodings = self.tokenizer.batch_encode_plus(X_test,
                                                          add_special_tokens=True,
                                                          return_attention_mask=True,
                                                          pad_to_max_length=True,
                                                          max_length=64,
                                                          return_tensors='pt')

        input_ids_test = test_encodings['input_ids']
        attention_masks_test = test_encodings['attention_mask']
        labels_test = torch.tensor(y_test)

        dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
        dataset_val = TensorDataset(input_ids_test, attention_masks_test, labels_test)

        return dataset_train, dataset_val, label_dict, len(possible_labels)

    def predict_data(self, file_path):
        # Read excel / read dataframe
        df = pd.read_excel(file_path)
        X = df['desc'].tolist()
        predict_encodings = self.tokenizer.batch_encode_plus(X,
                                                             return_attention_mask=True,
                                                             pad_to_max_length=True,
                                                             max_length=64,
                                                             return_tensors='pt')

        input_ids = predict_encodings['input_ids']
        attention_masks = predict_encodings['attention_mask']
        dataset_predict = TensorDataset(input_ids, attention_masks)

        dataloader_predict = DataLoader(dataset_predict,
                                        sampler=SequentialSampler(dataset_predict),
                                        batch_size=self.batch_size)
        return dataloader_predict, X

    def batch_data(self):
        # Prepare batch data
        dataloader_train = DataLoader(self.dataset_train,
                                      sampler=RandomSampler(self.dataset_train),
                                      batch_size=self.batch_size)
        dataloader_validation = DataLoader(self.dataset_val,
                                           sampler=SequentialSampler(self.dataset_val),
                                           batch_size=self.batch_size)
        return dataloader_train, dataloader_validation


class Dataloader_MDF2Rec:
    def __init__(self, model_name, bert_path):
        self.model_name = model_name
        self.bert_path = bert_path
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name, cache_dir=self.bert_path)
        self.dataset_train, self.dataset_val, self.label_dict, self.size = self.prepare_data()
        self.batch_size = 32
        self.dataloader_train, self.dataloader_validation = self.batch_data()

    def prepare_data(self):
        # Read excel / read dataframe
        df = pd.read_excel(r"data/recForm.xlsx")
        X = df['desc'].tolist()

        # Encoding the Labels
        possible_labels = df['recommendedPmForm'].unique()
        label_dict = {}
        for index, possible_label in enumerate(possible_labels):
            label_dict[possible_label] = index
        y = df['recommendedPmForm'].replace(label_dict).tolist()

        # Prepare training data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        # Tokenize and encode training and testing data
        train_encodings = self.tokenizer.batch_encode_plus(X_train,
                                                           add_special_tokens=True,
                                                           return_attention_mask=True,
                                                           pad_to_max_length=True,
                                                           max_length=64,
                                                           return_tensors='pt')

        input_ids_train = train_encodings['input_ids']
        attention_masks_train = train_encodings['attention_mask']
        labels_train = torch.tensor(y_train)

        test_encodings = self.tokenizer.batch_encode_plus(X_test,
                                                          add_special_tokens=True,
                                                          return_attention_mask=True,
                                                          pad_to_max_length=True,
                                                          max_length=64,
                                                          return_tensors='pt')

        input_ids_test = test_encodings['input_ids']
        attention_masks_test = test_encodings['attention_mask']
        labels_test = torch.tensor(y_test)

        dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
        dataset_val = TensorDataset(input_ids_test, attention_masks_test, labels_test)

        return dataset_train, dataset_val, label_dict, len(possible_labels)

    def predict_data(self, file_path):
        # Read excel / read dataframe
        df = pd.read_excel(file_path)
        X = df['desc'].tolist()
        predict_encodings = self.tokenizer.batch_encode_plus(X,
                                                             return_attention_mask=True,
                                                             pad_to_max_length=True,
                                                             max_length=64,
                                                             return_tensors='pt')

        input_ids = predict_encodings['input_ids']
        attention_masks = predict_encodings['attention_mask']
        dataset_predict = TensorDataset(input_ids, attention_masks)

        dataloader_predict = DataLoader(dataset_predict,
                                        sampler=SequentialSampler(dataset_predict),
                                        batch_size=self.batch_size)
        return dataloader_predict, X

    def batch_data(self):
        # Prepare batch data
        dataloader_train = DataLoader(self.dataset_train,
                                      sampler=RandomSampler(self.dataset_train),
                                      batch_size=self.batch_size)
        dataloader_validation = DataLoader(self.dataset_val,
                                           sampler=SequentialSampler(self.dataset_val),
                                           batch_size=self.batch_size)
        return dataloader_train, dataloader_validation


