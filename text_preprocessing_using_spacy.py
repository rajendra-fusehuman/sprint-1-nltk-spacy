# import all necessary libraries

import pandas as pd
import spacy
from nltk.stem import PorterStemmer
import re
from emoticons import EMOTICONS
from emojis import EMO_UNICODE
import time

# load an english model and initialize with an object call nlp
nlp = spacy.load("en_core_web_sm")


def lower_casing(text):
    """
    Lower case all the characters in the text.

    Parameters
    ----------
    text: string
        The raw text or string to be converted to lowercase

    Returns
    -------
    string
        The lowercase version of the input string
    """

    return text.lower()


def remove_punctuation(text):
    """
    Remove punctuation symbols from the given text using spaCy.

    Parameters
    ----------
    text: string
        The raw text from where to remove punctuation

    Returns
    -------
    string
        Text after all punctuations have been removed
    """

    doc = nlp(text)
    text_without_punct = " ".join([token.text for token in doc if not
                                   token.is_punct])

    return text_without_punct


def tokenize_words(text):
    """
    Tokenize the raw text splitted on the basis of words

    Parameters
    ----------
    text: string
        The raw text to tokenize

    Returns
    -------
    List
        List of words as tokens
    """

    doc = nlp(text)
    tokens = [token.text for token in doc]

    return tokens


def tokenize_sentence(text):
    """
    Tokenize the raw text splitted on the basis of sentences.

    Parameters
    ----------
    text: string
        The raw text to tokenize.

    Returns
    -------
    List
        List of sentences as tokens.
    """

    doc = nlp(text)
    sentences = list(doc.sents)

    return sentences


def remove_stopwords(text):
    """
    Remove stopwords from the given raw text.

    Parameters
    ----------
    text: String
        The raw text or string.

    Returns
    -------
    String
        Text with stopwords removed.
    """

    stopwords = nlp.Defaults.stop_words
    word_tokens = tokenize_words(text)
    text_without_sw = " ".join([token for token in word_tokens if token not in
                                stopwords])

    return text_without_sw


def remove_emoji(text):
    """
    Remove emoji from the given raw text, if there are any.
    Reference: https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304
    b

    Parameters
    ----------
    text: string
        The raw text.

    Returns
    -------
    String
        The text with emojis removed.
    """

    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)


def remove_emoticons(text):
    """
    Remove emoticons from the given raw text, if there are any.

    Parameters
    ----------
    text: string
        The raw text.

    Returns
    -------
    String
        The text with emoticons removed.
    """

    emoticon_pattern = re.compile("(" + "|".join(k for k in EMOTICONS) + ")")
    return emoticon_pattern.sub(r"", text)


def convert_emoji_to_words(text):
    """
    Convert emoji into words which can describe the emoji.

    Parameters
    ----------
    text: String
        The raw text.

    Returns
    -------
    String
        The text with emojis converted into string.
    """

    UNICODE_EMO = {v: k for k, v in EMO_UNICODE.items()}
    for emot in UNICODE_EMO:
        text = re.sub(
            r"(" + emot + ")",
            "_".join(UNICODE_EMO[emot].replace(",", "").replace(":", "")
                     .split()),
            text,
        )
    return text


def convert_emoticon_to_words(text):
    """
    Convert emoticon into words which can describe the emoticon.

    Parameters
    ----------
    text: String
        The raw text.

    Returns
    -------
    String
        The text with emoticons converted into string.
    """

    for emot in EMOTICONS:
        text = re.sub(
            "(" + emot + ")", "_".join(EMOTICONS[emot].replace(",", "")
                                       .split()),
            text
        )
    return text


def remove_urls(text):
    """
    Remove URLs present in the raw text.

    Parameters
    ----------
    text: String
        The raw text containing URLs.

    Returns
    -------
    String
        The text with URLs removed.
    """

    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    return re.sub(url_pattern, "", text)


def remove_html(text):
    """
    Remove HTML tags present in the raw text.

    Parameters
    ----------
    text: String
        The raw text containing HTML tags.

    Returns
    -------
    String
        The text with HTML tags removed.
    """

    html_pattern = re.compile("<.*?>")
    return re.sub(html_pattern, "", text)


def stemming(text):
    """
    Perform stemming on the given raw text utilizing PorterStemmer from NLTK.

    Note: spaCy doesn't directly support stemming, as it is primarily focused
    on lemmatization. However, we can still perform stemming using spaCy by
    combining it with NLTK.

    Parameters
    ----------
    text: String
        Raw text from the dataframe

    Returns
    -------
    String
        The text with words reduced to their respective stems whenever
        possible
    """

    ps = PorterStemmer()
    tokens = tokenize_words(text)
    text_with_stem = " ".join([ps.stem(token) for token in tokens])

    return text_with_stem


def lemmatization(text):
    """
    Perform lemmatization on the given raw text utilizing .lemma_ attribute
    of the spacy.tokens.token.Token object.

    Parameters
    ----------
    text: String
        Raw text from the dataframe.

    Returns
    -------
    String
        The text with words reduced to their respective lemmas whenever
        possible.
    """

    doc = nlp(text)
    text_with_lemmas = " ".join([token.lemma_ for token in doc])

    return text_with_lemmas


def get_pos_tag(text):
    """
    Return the POS tag of each word from the given raw text.

    Parameters
    ----------
    text: String
        The raw text.

    Returns
    -------
    List of tuples
        List of tuples with each tuple containing the token text, its pos
        attribute and tag attribute.
    """

    doc = nlp(text)
    pos_list = [(token.text, token.pos_, token.tag_) for token in doc]

    return pos_list


def named_entity_recognition(text):
    """
    Return the entity label of each name present in the given text.

    Parameters
    ----------
    text: String
        The raw text containing names whose entity label is to be found out.

    Returns
    -------
    List of tuples
        List of tuples with each tuple containing the entity text and its
        corresponding entity label
    """

    doc = nlp(text)
    ner_list = [(entity.text, entity.label_) for entity in doc.ents]

    return ner_list


def preprocess_text(text):
    text = text.replace("\n", " ")  # removing next line
    text = lower_casing(text)
    text = remove_urls(text)
    text = remove_html(text)
    text = convert_emoji_to_words(text)
    text = convert_emoticon_to_words(text)
    text = remove_stopwords(text)
    text = lemmatization(text)
    text = remove_punctuation(text)
    # remove all characters except alphabets and numbers
    # text = re.sub(r"[^a-zA-Z0-9 ]+", "", text)
    text = re.sub("( . )", " ", text)  # removing a single character word
    text = re.sub(r"\s+", " ", text)  # removing multiple white spaces
    text = text.strip()

    return text


if __name__ == "__main__":
    start = time.time()
    df = pd.read_csv("dataset/sample_dataset_real_or_fake_news.csv")
    df["preprocessed_text"] = df["text"].apply(lambda x: preprocess_text(x))
    df.drop(columns=['title', 'text', 'label'], inplace=True)
    df.to_csv("outputs/preprocessed-text-with-spacy.csv", index=False)
    end = time.time()
    print(f"Time Elapsed: {end-start:.4f} seconds")
