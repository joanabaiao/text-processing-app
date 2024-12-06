from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from spacy.displacy import render
from typing import List

# from src.config import *
MODEL_NAME_DISTILBERT: str = "elastic/distilbert-base-cased-finetuned-conll03-english"
MODEL_PATH_DISTILBERT: str = "models/DistilBERT/model"
TOKENIZER_PATH_DISTILBERT: str = "models/DistilBERT/tokenizer"


class NERModel:

    def __init__(self, model_name, model_path, tokenizer_path):

        self.model_name = model_name

        print(f"Loading {model_name} tokenizer and model...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.id2label = self.model.config.id2label

        print("Model loaded!")

    def get_ner_results(self, text):

        pipe = pipeline(
            model=self.model_name,
            tokenizer=self.tokenizer,
            task="ner",
            aggregation_strategy="simple",
        )

        results = pipe(text)

        return results

    def visualize_results(self, text):

        results = self.get_ner_results(text)

        entities = []
        for output in results:
            entry = {}
            entry["start"] = output["start"]
            entry["end"] = output["end"]
            entry["label"] = output["entity_group"]
            entities.append(entry)

        render_data = [{"text": text, "ents": entities, "title": None}]
        html = render(render_data, style="ent", manual=True, page=False)

        return html


if __name__ == "__main__":

    text = "Emma Watson visited London last weekend to promote her new movie."
    model = NERModel(
        MODEL_NAME_DISTILBERT, MODEL_PATH_DISTILBERT, TOKENIZER_PATH_DISTILBERT
    )
    results = model.get_ner_results(text)
    print(results)
