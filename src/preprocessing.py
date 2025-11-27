import re
import spacy

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("sentencizer")

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_sentences(text):
    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if len(s.text.strip()) > 10]
