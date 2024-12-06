from transformers import BartForConditionalGeneration, BartTokenizer, AutoTokenizer

# from ..config import *

MODEL_NAME_BART: str = "facebook/bart-large-cnn"
MODEL_PATH_BART: str = "models/BART/tokenizer"
TOKENIZER_PATH_BART: str = "models/BART/tokenizer"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_BART)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME_BART)

tokenizer.save_pretrained(TOKENIZER_PATH_BART)
model.save_pretrained(MODEL_PATH_BART)

print("Saved!")
