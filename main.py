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

MODEL_TYPE = "Desc2MDF" # MDF2Rec / Desc2MDF
IS_TRAIN_MODEL_NEEDED = False
IS_INPUT_BY_CSV = False
INPUT_BY_TEXT = ['INFUSION PUMPS, LONGG', 'WATER PURIFICATION']

model_name = 'bert-base-uncased'
bert_path = ''

match (MODEL_TYPE):
    case ("MDF2Rec"):
        data = dl.Dataloader_MDF2Rec(model_name, bert_path)
    case ("Desc2MDF"):
        data = dl.Dataloader_Desc2MDF(model_name, bert_path)

model = BertClassifier(model_name, MODEL_TYPE, data)

if(IS_TRAIN_MODEL_NEEDED and IS_INPUT_BY_CSV):
    model._train(data.dataloader_train, data.dataloader_validation)

file_path = "data/beseformDB.mdfAssetDescriptions.xlsx"
dataloader_predict, predict_ids = data.predict_data(file_path, IS_INPUT_BY_CSV, INPUT_BY_TEXT)
prediction = model._predict(dataloader_predict, predict_ids)
print(prediction)

print('Hello World')
