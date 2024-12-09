import nltk
from keybert import KeyBERT
from nltk.corpus import wordnet
from typing import List, Dict


class KeywordModel:

    def __init__(self):
        """Initialize the KeywordModel with KeyBERT."""
        self.model = KeyBERT()
        self.keywords = []
        self.keywords_synonyms = {}

    def extract_keywords(self, text: str, n_keywords: int) -> List[str]:
        """
        Extract keywords from the input text using KeyBERT.
        """
        try:
            keywords_extracted = self.model.extract_keywords(
                text, keyphrase_ngram_range=(1, 1), top_n=n_keywords
            )
            self.keywords = [word[0] for word in keywords_extracted]

        except Exception as e:
            print(f"Error extracting keywords: {e}")
            self.keywords = []

        return self.keywords

    def get_synonyms_for_keyword(self, keyword: str, n_synonyms: int = 5) -> List[str]:
        """
        Retrieve synonyms for a given keyword using WordNet.
        """
        try:
            synonyms = []
            for syn in wordnet.synsets(keyword):
                for l in syn.lemmas():
                    synonyms.append(l.name().replace("_", " "))

            synonyms = list(set(synonyms))
            synonyms_filtered = [
                syn for syn in synonyms if syn.lower() != keyword.lower()
            ]
            synonyms_filtered = synonyms_filtered[0:n_synonyms]

        except Exception as e:
            print(f"Error retrieving synonyms for {keyword}: {e}")
            synonyms_filtered = []

        return synonyms_filtered

    def get_all_synonyms(self, n_synonyms: int) -> Dict[str, List[str]]:
        """
        Retrieve synonyms for a list of keywords.
        """
        self.keywords_synonyms = {}
        for k in self.keywords:
            synonyms = self.get_synonyms_for_keyword(k, n_synonyms=n_synonyms)
            self.keywords_synonyms[k] = synonyms

        return self.keywords_synonyms

    def extract_keywords_and_synonyms(
        self, text: str, n_keywords: int, n_synonyms: int
    ) -> Dict[str, List[str]]:
        """
        Extract keywords and their synonyms from the input text.
        """

        keywords = self.extract_keywords(text, n_keywords=n_keywords)
        keyword_synonyms = self.get_all_synonyms(n_synonyms=n_synonyms)

        return keyword_synonyms


################################################################


if __name__ == "__main__":

    sample_text = """
    Machine learning is a method of data analysis that automates analytical model building. 
    It is a branch of artificial intelligence based on the idea that systems can learn from data, 
    identify patterns, and make decisions with minimal human intervention.
    """

    keyword_model = KeywordModel()

    # Extract keywords
    print("Extracting keywords...")
    keywords = keyword_model.extract_keywords(sample_text, n_keywords=5)
    # print("Default Keywords:", keywords)

    # Get synonyms
    print("\nGetting synonyms for each keyword...")
    keywords_synonyms = keyword_model.get_all_synonyms(n_synonyms=5)

    for keyword, synonyms in keywords_synonyms.items():
        print(f"Keyword: {keyword}")
        print(f"Synonyms: {synonyms}\n")

    ###### ALL IN ONE STEP ######
    keywords_with_synonyms = keyword_model.extract_keywords_and_synonyms(
        sample_text, n_keywords=5, n_synonyms=3
    )
    print(keywords_with_synonyms)
