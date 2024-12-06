from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
from sentence_splitter import SentenceSplitter

# from src.config import *
MODEL_NAME_PEGASUS: str = "tuner007/pegasus_paraphrase"
MODEL_PATH_PEGASUS: str = "models/Pegasus/model"
TOKENIZER_PATH_PEGASUS: str = "models/Pegasus/tokenizer"


class ParaphrasingModel:

    def __init__(self, model_name, model_path, tokenizer_path):

        self.model_name = model_name
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = PegasusTokenizer.from_pretrained(tokenizer_path)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_path).to(
            torch_device
        )
        self.splitter = SentenceSplitter(language="en")

        print("Model loaded!")

    def paraphrase_sentence(self, sentence):

        inputs = self.tokenizer([sentence], return_tensors="pt", truncation=True)
        paraphrase_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_beams=4,
            max_length=128,
            early_stopping=True,
        )

        paraphrased_sentence = self.tokenizer.decode(
            paraphrase_ids[0], skip_special_tokens=True
        )

        return paraphrased_sentence

    def paraphrase_text(self, text):

        sentence_list = self.splitter.split(text)
        results = []

        for sentence in sentence_list:
            sentence = sentence.strip()
            paraphrased_sentence = self.paraphrase_sentence(sentence)
            results.append(paraphrased_sentence)

        paraphrased_text = " ".join(results)

        return paraphrased_text


if __name__ == "__main__":
    text = "The site of the Dolomites comprises a mountain range in the northern Italian Alps, numbering 18 peaks which rise to above 3,000 metres and cover 141,903 ha. It features some of the most beautiful mountain landscapes anywhere, with vertical walls, sheer cliffs and a high density of narrow, deep and long valleys. A serial property of nine areas that present a diversity of spectacular landscapes of international significance for geomorphology marked by steeples, pinnacles and rock walls, the site also contains glacial landforms and karst systems. It is characterized by dynamic processes with frequent landslides, floods and avalanches. The property also features one of the best examples of the preservation of Mesozoic carbonate platform systems, with fossil records."
    model = ParaphrasingModel(
        MODEL_NAME_PEGASUS, MODEL_PATH_PEGASUS, TOKENIZER_PATH_PEGASUS
    )

    summary = model.paraphrase_text(text)
    print(summary)
