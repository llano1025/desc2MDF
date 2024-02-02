from data import data_loader as dl
from exc.exc import BertClassifier
import os

# Use environment
proxy = ''
os.environ['http_proxy'] = proxy
os.environ['HTTP_PROXY'] = proxy
os.environ['https_proxy'] = proxy
os.environ['HTTPS_PROXY'] = proxy
os.environ['CURL_CA_BUNDLE'] = ''

model_name = 'bert-base-uncased'
bert_path = ''
model_type = "MDF2Rec"  # MDF2Rec / Desc2MDF

# data = dl.Dataloader_Desc2MDF(model_name, bert_path)
data = dl.Dataloader_MDF2Rec(model_name, bert_path)
model = BertClassifier(model_name, model_type, data)
model._train(data.dataloader_train, data.dataloader_validation)

file_path = "data/beseformDB.mdfAssetDescriptions.xlsx"
dataloader_predict = data.predict_data(file_path)
prediction = model._predict(dataloader_predict)
print(prediction)

print('Hello World')
