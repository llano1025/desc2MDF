from data import data_loader as dl
from exc.exc import BertClassifier
from flask import Flask, jsonify, request
import os
import ast

# Use environment
proxy = ''
os.environ['http_proxy'] = proxy
os.environ['HTTP_PROXY'] = proxy
os.environ['https_proxy'] = proxy
os.environ['HTTPS_PROXY'] = proxy
os.environ['CURL_CA_BUNDLE'] = ''

MODEL_TYPE = "MDF2Rec" # MDF2Rec / Desc2MDF
IS_TRAIN_MODEL_NEEDED = False
IS_INPUT_OUTPUT_BY_CSV = True

model_name = 'bert-base-uncased'
bert_path = ''
parsed_list = []

match (MODEL_TYPE):
    case ("MDF2Rec"):
        data = dl.Dataloader_MDF2Rec(model_name, bert_path)
    case ("Desc2MDF"):
        data = dl.Dataloader_Desc2MDF(model_name, bert_path)

model = BertClassifier(model_name, MODEL_TYPE, data)

if(IS_TRAIN_MODEL_NEEDED and IS_INPUT_OUTPUT_BY_CSV):
    model._train(data.dataloader_train, data.dataloader_validation)

file_path = "data/beseformDB.mdfAssetDescriptions.xlsx"
dataloader_predict, predict_ids = data.predict_data(file_path, IS_INPUT_OUTPUT_BY_CSV, parsed_list)
prediction = model._predict(dataloader_predict, predict_ids, IS_INPUT_OUTPUT_BY_CSV)
print(prediction)

app = Flask(__name__)

@app.route("/", methods=['POST'])
def getRecAndMDF():
    IS_INPUT_OUTPUT_BY_CSV = False
    data = request.json
    parsed_list = ast.literal_eval(data['desc'])

    match (MODEL_TYPE):
        case ("MDF2Rec"):
            data = dl.Dataloader_MDF2Rec(model_name, bert_path)
        case ("Desc2MDF"):
            data = dl.Dataloader_Desc2MDF(model_name, bert_path)
    model = BertClassifier(model_name, MODEL_TYPE, data)
    dataloader_predict, predict_ids = data.predict_data(file_path, IS_INPUT_OUTPUT_BY_CSV, parsed_list)
    api_prediction = model._predict(dataloader_predict, predict_ids, IS_INPUT_OUTPUT_BY_CSV)
    return jsonify({"recommended_mdf": api_prediction})

print('Hello World')
