"""
NAN.ai after sales chatbot
This transforms a user utterance into a machine understandable units
i.e. a predefined semantic frame as a precursor to intent identification
This model is integrated to the NAN.ai ecosystem via web service

This script implements pre-processing
"""

import spacy
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

nlp = spacy.load("en_core_web_sm")

# Removes custom stopwords in English and Filipino
def remove_stopwords(stoplist_tl, line):
    
    # Add to default stoplist
    nlp.Defaults.stop_words |= stoplist_tl

    line = line.lower()
    line = " ".join(token.lemma_ for token in nlp(line) if not token.is_stop)
    line = " ".join(token.lemma_ for token in nlp(line) if not token.is_punct and not token.is_digit)

    return line

def stemming(line):

    tokenized = word_tokenize(line)
    stemmed = []

    for t in tokenized:
        stemmed.append(ps.stem(t))
        print(t, " : ", ps.stem(t))

    return stemmed


def encoding_doc(token, words):
  return(token.texts_to_sequences(words))


def padding_doc(encoded_doc, max_length):
  return(pad_sequences(encoded_doc, maxlen = max_length, padding = "post"))