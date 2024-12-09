import streamlit as st
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


@st.cache_resource
def load_models():

    # KEYBERT
    print("Loading KeyBERT model...")
    keyword_model = KeywordModel()
    print("Loaded!")

    # DISTILBERT
    print("\nLoading DistilBERT model...")
    ner_model = NERModel(
        MODEL_NAME_DISTILBERT, MODEL_NAME_DISTILBERT, MODEL_NAME_DISTILBERT
    )
    print("Loaded!")

    # BART
    print("\nLoading BART model...")
    sum_model = SummarizationModel(MODEL_NAME_BART, MODEL_NAME_BART, MODEL_NAME_BART)
    print("Loaded!")

    print("\nLoading Pegasus model...")
    para_model = ParaphrasingModel(
        MODEL_NAME_PEGASUS, MODEL_NAME_PEGASUS, MODEL_NAME_PEGASUS
    )
    print("Loaded!")

    return keyword_model, ner_model, sum_model, para_model


keyword_model, ner_model, sum_model, para_model = load_models()

################################################################################################

# Display the image
# image1 = Image.open("docs/recipe.jpg")
# st.image(image1)

st.title("Text Processing API")

st.write(
    "This application leverages advanced NLP techniques "
    "using Hugging Face Transformers to provide a comprehensive analysis of your input text. \n"
)
st.write(
    "It can **summarize**, **paraphrase**, identify **named entities**, and extract **keywords** along with their synonyms, all at once."
    "Simply enter your text below, and the app will handle the rest!"
)

tasks = st.multiselect(
    "Select tasks to perform",
    [
        "Summarization",
        "Paraphrasing",
        "Keywords",
        "Named entity recognition (NER)",
    ],  # List of tasks
    default=["Summarization"],  # Default selected task(s)
)

input_text = st.text_area("Enter your text here:")

if st.button("Process"):

    ###################################################
    # SUMMARIZATION
    ###################################################
    if "Summarization" in tasks:

        if input_text:
            # Summarize Text
            # summary = summarizer(input_text, max_length=130, min_length=30, do_sample=False)
            st.subheader("Summary:")
            with st.spinner("Processing..."):
                summary = sum_model.summarize_text(input_text)
                st.write(summary)
        else:
            st.warning("Please enter text to summarize.")

    ###################################################
    # PARAPHRASING
    ###################################################
    if "Paraphrasing" in tasks:
        if input_text:
            st.subheader("Paraphrased text:")
            with st.spinner("Processing..."):
                paraphrased_text = para_model.paraphrase_text(input_text)
                st.write(paraphrased_text)
        else:
            st.warning("Please enter text to paraphrase.")

    ###################################################
    # KEYWORDS AND SYNONYMS
    ###################################################
    if "Keywords" in tasks:
        if input_text:
            st.subheader("Keywords:")
            with st.spinner("Processing..."):
                keywords_with_synonyms = keyword_model.extract_keywords_and_synonyms(
                    input_text, n_synonyms=3, n_keywords=6
                )
                cols = st.columns(3)
                for idx, (keyword, synonyms) in enumerate(
                    keywords_with_synonyms.items()
                ):
                    col = cols[idx % 3]
                    with col:
                        with st.expander(f"**{keyword}**"):
                            # Display synonyms as a bullet list
                            st.markdown("\n".join([f"- {syn}" for syn in synonyms]))
        else:
            st.warning("Please enter text to extract keywords.")

    ###################################################
    # NAMED ENTITY RECOGNITION
    ###################################################
    if "Named entity recognition (NER)" in tasks:
        if input_text:
            # Named Entity Recognition
            st.subheader("Named Entities:")
            with st.spinner("Processing..."):
                ner_html = ner_model.visualize_results(input_text)
                st.html(ner_html)
        else:
            st.warning("Please enter text for NER.")
