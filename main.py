from data import data_loader as dl
from exc.exc import BertClassifier


# Use environment
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# proxy = 'http://18366:@N37f343734@!@proxy.emsd.hksarg:80'
# os.environ['http_proxy'] = proxy
# os.environ['HTTP_PROXY'] = proxy
# os.environ['https_proxy'] = proxy
# os.environ['HTTPS_PROXY'] = proxy
# os.environ['CURL_CA_BUNDLE'] = ''

model_name = 'bert-base-uncased'
bert_path = 'G:/huggingface/hub'
model_type = "Desc2MDF"
data = dl.Dataloader_Desc2MDF(model_name, bert_path)
model = BertClassifier(model_name, model_type, data)
# model._train(data.dataloader_train, data.dataloader_validation)


file_path = "data/beseformDB.mdfAssetDescriptions.xlsx"
pred_data = data.predict_data(file_path)
prediction = model._predict(pred_data)

print('hello world')
