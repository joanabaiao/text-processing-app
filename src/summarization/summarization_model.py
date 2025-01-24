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

    def summarize_text(self, text, length_ratio=0.7, threshold_length=20):

        input_length = len(text.split())
        if input_length < threshold_length:
            return text

        # min_length = max(10, int(input_length * length_ratio * 0.5))
        max_summary_length = max(threshold_length, int(input_length * length_ratio))
        inputs = self.tokenizer([text], return_tensors="pt")

        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=max_summary_length,
            # max_length=128,
            # min_length=min_length,
            length_penalty=1,
            num_beams=4,
            early_stopping=True,
        )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary


if __name__ == "__main__":
    text = "Trump has also ordered the declassification of files relating to the deaths of John F Kennedy, Robert Kennedy and Martin Luther King Jr as part of another flurry of executive orders."
    # text = "today is sunny"
    model = SummarizationModel(MODEL_NAME_BART, MODEL_PATH_BART, TOKENIZER_PATH_BART)
    summary = model.summarize_text(text)

    print(summary)
