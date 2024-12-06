from transformers import AutoTokenizer, AutoModelForTokenClassification

# from ..config import *

MODEL_NAME_DISTILBERT: str = "elastic/distilbert-base-cased-finetuned-conll03-english"
MODEL_PATH_DISTILBERT: str = "models/DistilBERT/model"
TOKENIZER_PATH_DISTILBERT: str = "models/DistilBERT/tokenizer"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_DISTILBERT)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME_DISTILBERT)

tokenizer.save_pretrained(TOKENIZER_PATH_DISTILBERT)
model.save_pretrained(MODEL_PATH_DISTILBERT)
