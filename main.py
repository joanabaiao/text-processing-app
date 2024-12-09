import ast
from PIL import Image
import re
import pandas as pd
from nltk.stem import WordNetLemmatizer

from src.keywords.keywords_model import KeywordModel
from src.ner.ner_model import NERModel
from src.summarization.summarization_model import SummarizationModel
from src.paraphrasing.paraphrasing_model import ParaphrasingModel

from src.config import *

################################################################################################


def main(selected_tasks, input_text):
    print("Processing the text...")

    # Execute tasks based on user selection
    if "Summarization" in selected_tasks:

        sum_model = SummarizationModel(
            MODEL_NAME_BART, MODEL_PATH_BART, TOKENIZER_PATH_BART
        )

        print("\n--- Summarization ---")
        summary = sum_model.summarize_text(input_text)
        print(f"Summary: {summary}")

    if "Paraphrasing" in selected_tasks:

        para_model = ParaphrasingModel(
            MODEL_NAME_PEGASUS, MODEL_PATH_PEGASUS, TOKENIZER_PATH_PEGASUS
        )

        print("\n--- Paraphrasing ---")
        paraphrased_text = para_model.paraphrase_text(input_text)
        print(f"Paraphrased Text: {paraphrased_text}")

    if "Keywords" in selected_tasks:

        keyword_model = KeywordModel()

        print("\n--- Keywords and Synonyms ---")
        keywords_with_synonyms = keyword_model.extract_keywords_and_synonyms(
            input_text, n_synonyms=3, n_keywords=6
        )
        for keyword, synonyms in keywords_with_synonyms.items():
            print(f"Keyword: {keyword}")
            print(f"Synonyms: {', '.join(synonyms)}")

    if "Named entity recognition" in selected_tasks:

        ner_model = NERModel(
            MODEL_NAME_DISTILBERT, MODEL_PATH_DISTILBERT, TOKENIZER_PATH_DISTILBERT
        )

        print("\n--- Named Entity Recognition ---")
        ner_html = ner_model.get_ner_results(input_text)
        print(f"NER Output:\n{ner_html}")

    print("\nProcessing complete. Thank you for using the Text Processing API!")


if __name__ == "__main__":

    selected_tasks = [
        # "Summarization",
        # "Paraphrasing",
        "Keywords",
        # "Named entity recognition",
    ]

    input_text = "Cats and dogs are among the most popular pets worldwide, each offering unique qualities that appeal to different owners. Cats are known for their independence, agility, and curious nature. They are often quiet companions who enjoy lounging in sunny spots or playfully chasing after toys. On the other hand, dogs are celebrated for their loyalty, energy, and ability to form strong bonds with humans. They thrive on companionship and require regular exercise, making them great for active families. Both animals have distinct communication styles. Cats often purr to express contentment or meow for attention, while dogs bark or wag their tails to communicate emotions. Despite their differences, cats and dogs can coexist harmoniously in many households with proper training and care. Whether you prefer the playful antics of a dog or the serene presence of a cat, both pets bring joy and companionship to millions of homes."

    main(selected_tasks, input_text)
