from data import data_loader as dl
from exc.exc import BertClassifier
from flask import Flask, jsonify, request
from utils.metrics import parse_to_result
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
IS_INPUT_OUTPUT_BY_CSV = False

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

# API
app = Flask(__name__)
@app.route("/", methods=['POST'])
def getRecAndMDF():
    IS_INPUT_OUTPUT_BY_CSV = False
    data = request.json
    parsed_list = ast.literal_eval(data['desc'])
    # Predict MDF2Rec 
    MDF2Rec_data = dl.Dataloader_MDF2Rec(model_name, bert_path)
    MDF2Rec_model = BertClassifier(model_name, 'MDF2Rec', MDF2Rec_data)
    MDF2Rec_dataloader_predict, MDF2Rec_predict_ids = MDF2Rec_data.predict_data(file_path, IS_INPUT_OUTPUT_BY_CSV, parsed_list)
    MDF2Rec_prediction = MDF2Rec_model._predict(MDF2Rec_dataloader_predict, MDF2Rec_predict_ids, IS_INPUT_OUTPUT_BY_CSV)
    # Predict Desc2MDF 
    Desc2MDF_data = dl.Dataloader_Desc2MDF(model_name, bert_path)
    Desc2MDF_model = BertClassifier(model_name, 'Desc2MDF', Desc2MDF_data)
    Desc2MDF_dataloader_predict, Desc2MDF_predict_ids = Desc2MDF_data.predict_data(file_path, IS_INPUT_OUTPUT_BY_CSV, parsed_list)
    Desc2MDF_prediction = Desc2MDF_model._predict(Desc2MDF_dataloader_predict, Desc2MDF_predict_ids, IS_INPUT_OUTPUT_BY_CSV)

    result = parse_to_result(parsed_list, MDF2Rec_prediction, Desc2MDF_prediction)

    return jsonify({"data": result})
