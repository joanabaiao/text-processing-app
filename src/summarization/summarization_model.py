from transformers import AutoTokenizer, BartForConditionalGeneration, BartTokenizer
import textwrap

# from src.config import *
MODEL_NAME_BART: str = "facebook/bart-large-cnn"
MODEL_PATH_BART: str = "models/BART/model"
TOKENIZER_PATH_BART: str = "models/BART/tokenizer"


class SummarizationModel:

    def __init__(self, model_name, model_path, tokenizer_path):

        self.model_name = model_name
        print(f"Loading {model_name} tokenizer and model...")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = BartForConditionalGeneration.from_pretrained(model_path)

        print("Model loaded!")

    def summarize_text(self, text, length_ratio=0.7):

        input_length = len(text.split())
        # min_length = max(10, int(input_length * length_ratio * 0.5))
        # max_lenght = max(128, int(input_length * length_ratio))
        inputs = self.tokenizer([text], return_tensors="pt")

        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=128,
            # max_length=max_lenght,
            # min_length=min_length,
            length_penalty=2,
            num_beams=4,
            early_stopping=True,
        )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        print(
            f"Input length: {input_length} words -> Summary length: {len(summary.split())} words"
        )

        return summary


if __name__ == "__main__":
    # text = "The quick brown fox jumps over the lazy white dog quickly, and the fox seems very agile, clever and fun."
    text = "The site of the Dolomites comprises a mountain range in the northern Italian Alps, numbering 18 peaks which rise to above 3,000 metres and cover 141,903 ha. It features some of the most beautiful mountain landscapes anywhere, with vertical walls, sheer cliffs and a high density of narrow, deep and long valleys. A serial property of nine areas that present a diversity of spectacular landscapes of international significance for geomorphology marked by steeples, pinnacles and rock walls, the site also contains glacial landforms and karst systems. It is characterized by dynamic processes with frequent landslides, floods and avalanches. The property also features one of the best examples of the preservation of Mesozoic carbonate platform systems, with fossil records."
    model = SummarizationModel(MODEL_NAME_BART, MODEL_PATH_BART, TOKENIZER_PATH_BART)
    summary = model.summarize_text(text)

    print(summary)
