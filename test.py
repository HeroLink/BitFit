
from transformers import AutoModelForSequenceClassification, AutoConfig, DebertaForSequenceClassification, DistilBertForSequenceClassification


# model_name = 'distilbert-base-uncased'
model_name = 'microsoft/deberta-base'
config = AutoConfig.from_pretrained(model_name, num_labels=2, return_dict=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
# model = DistilBertForSequenceClassification.from_pretrained(model_name, return_dict=True).distilbert
# model = DebertaForSequenceClassification.from_pretrained(model_name, return_dict=True).deberta
# print(model)
# print(model.deberta)

for name, param in model.named_parameters():
    print(name)