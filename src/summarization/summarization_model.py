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
    text = "Trump has also ordered the declassification of files relating to the deaths of John F Kennedy, Robert Kennedy and Martin Luther King Jr as part of another flurry of executive orders."
    # text = "The quick brown fox jumps over the lazy white dog quickly, and the fox seems very agile, clever and fun."

    text = """In recent years, the rapid development of technology has drastically changed the way people live, work, and interact with one another. The advent of smartphones, social media, and other digital tools has transformed how individuals communicate and access information. People are now more connected than ever before, with the ability to communicate instantly across vast distances, share content at the touch of a button, and stay informed about global events in real-time. However, with these advancements come new challenges, such as concerns over privacy, the spread of misinformation, and the impact of technology on mental health.

One of the most significant developments has been the rise of social media platforms. These platforms have redefined how people interact with one another, allowing individuals to form virtual communities, share their thoughts and opinions, and engage in discussions on a wide range of topics. While social media has provided a space for people to connect and share experiences, it has also been criticized for fostering negativity, cyberbullying, and the spread of fake news. The increasing use of algorithms to curate content has also raised concerns about filter bubbles, where users are only exposed to content that aligns with their existing beliefs and opinions, further dividing society.

Another area where technology has had a profound impact is in the workplace. The rise of remote work and the use of digital tools has enabled people to work from virtually anywhere, providing greater flexibility and opportunities for individuals to balance their professional and personal lives. However, this shift has also created challenges, such as the blurring of boundaries between work and home life, leading to increased stress and burnout. Additionally, the rise of automation and artificial intelligence has raised questions about the future of work and the potential for job displacement in certain industries.

Despite these challenges, the overall impact of technology on society has been overwhelmingly positive. Advances in fields such as medicine, education, and transportation have improved the quality of life for many people, enabling individuals to live longer, healthier lives, and access new opportunities. The key to navigating the future of technology lies in finding ways to harness its potential while addressing its negative aspects. By fostering open dialogue, promoting digital literacy, and implementing thoughtful policies, society can ensure that technology continues to serve as a force for good in the world.
"""
    model = SummarizationModel(MODEL_NAME_BART, MODEL_PATH_BART, TOKENIZER_PATH_BART)
    summary = model.summarize_text(text)

    print(summary)
