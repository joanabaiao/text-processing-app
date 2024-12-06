from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# from ..config import *

MODEL_NAME_PEGASUS: str = "tuner007/pegasus_paraphrase"
MODEL_PATH_PEGASUS: str = "models/Pegasus/model"
TOKENIZER_PATH_PEGASUS: str = "models/Pegasus/tokenizer"

tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME_PEGASUS)
model = PegasusForConditionalGeneration.from_pretrained(MODEL_NAME_PEGASUS)

tokenizer.save_pretrained(TOKENIZER_PATH_PEGASUS)
model.save_pretrained(MODEL_PATH_PEGASUS)
