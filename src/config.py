#####################################
# Name Entity Recognition (NER) - DistilBERT
#####################################
MODEL_NAME_DISTILBERT: str = "elastic/distilbert-base-cased-finetuned-conll03-english"
MODEL_PATH_DISTILBERT: str = "models/DistilBERT/model"
TOKENIZER_PATH_DISTILBERT: str = "models/DistilBERT/tokenizer"

#####################################
# SUMMARIZATION - BART
#####################################
MODEL_NAME_BART: str = "facebook/bart-large-cnn"
MODEL_PATH_BART: str = "models/BART/tokenizer"
TOKENIZER_PATH_BART: str = "models/BART/tokenizer"

#####################################
# PARAPHRASING - Pegasus
#####################################
MODEL_NAME_PEGASUS: str = "tuner007/pegasus_paraphrase"
MODEL_PATH_PEGASUS: str = "models/Pegasus/model"
TOKENIZER_PATH_PEGASUS: str = "models/Pegasus/tokenizer"
