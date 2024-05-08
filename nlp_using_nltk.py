# import all the necessary libraries
import nltk
import re
import string
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from emoticons import EMOTICONS
from emojis import EMO_UNICODE

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")


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
    Remove punctuation symbols from the given text.

    Parameters
    ----------
    text: string
        The raw text from where to remove punctuation

    Returns
    -------
    string
        Text after all punctuations have been removed
    """

    punctuations = string.punctuation
    text = text.translate(str.maketrans("", "", punctuations))
    return text


def tokenize_words(text):
    """
    Tokenize the raw text

    Parameters
    ----------
    text: string
        The raw text to tokenize

    Returns
    -------
    List
        List of words as tokens
    """

    tokens = word_tokenize(text)
    return tokens


def tokenize_sentence(text):
    """
    Tokenize the raw text

    Parameters
    ----------
    text: string
        The raw text to tokenize

    Returns
    -------
    List
        List of sentences as tokens
    """

    tokens = sent_tokenize(text)
    return tokens


def remove_stopwords(text):
    """
    Remove stopwords for the given raw text.

    Parameters
    ----------
    text: String
        The raw text or string.

    Returns
    -------
    String
        Text with stopwords removed.
    """

    stop_words = stopwords.words("english")
    # doesn't --> doesnt
    new_stop = [re.sub("[^a-z]", "", word) for word in stop_words]
    stop_words.extend(new_stop)
    # extra stop words not present in nltk corpus and added manually
    extra_stopwords = [
        "'s",
        "'t",
        "'m",
        "'d",
        "'ll",
        "'o",
        "'re",
        "'ve",
        "'y",
        "didn't",
        "didn",
    ]
    stop_words.extend(extra_stopwords)
    stop_words = list(set(stop_words))  # removing duplicates
    word_tokens = word_tokenize(text)
    text_without_sw = " ".join([word for word in word_tokens if word not in
                                stop_words])

    return text_without_sw


def remove_emoji(text):
    """
    Remove emoji from the given raw text, if there are any.
    Reference: https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304
    b

    Parameters
    ----------
    text: string
        The raw text

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
        The raw text

    Returns
    -------
    String
        The text with emojis removed.
    """

    emoticon_pattern = re.compile("(" + "|".join(k for k in EMOTICONS) + ")")
    return emoticon_pattern.sub(r"", text)


def convert_emoji_to_words(text):
    """
    Convert emoji into words to describe the emoji

    Parameters
    ----------
    text: String
        The raw text

    Returns
    -------
    String
        The text with emojis converted into string
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
    Convert emoticon into words to describe the emoticon

    Parameters
    ----------
    text: String
        The raw text

    Returns
    -------
    String
        The text with emoticons converted into string
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
    Remove URL present in the raw text.

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
    Perform stemming on the given raw text utilizing PorterStemmer from NLTK

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
    tokens = word_tokenize(text)
    text_with_stem = " ".join([ps.stem(token) for token in tokens])

    return text_with_stem


def lemmatization(text):
    """
    Perform lemmatization on the given raw text utilizing WordNetLemmatizer
    from NLTK without taking into consideration the POS tag of the word/token.

    Parameters
    ----------
    text: String
        Raw text from the dataframe

    Returns
    -------
    String
        The text with words reduced to their respective lemmas whenever
        possible
    """

    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    text_with_lemma = " ".join([lemmatizer.lemmatize(token) for token in
                                tokens])

    return text_with_lemma


def _get_pos_tag(word):
    """
    Return the POS tag of the word based on wordnet vocab

    Parameters
    ----------
    word: String
        The word whose POS tag is to be found out.

    Returns
    -------
    String
        POS tag of the word. By default the POS tag is recognized as a noun.
    """

    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
    }

    return tag_dict.get(tag, wordnet.NOUN)


def lemmatization_with_pos(text):
    """
    Perform lemmatization on the given raw text utilizing WordNetLemmatizer
    from NLTK taking into consideration the POS tag of the word.

    Parameters
    ----------
    text: String
        Raw text from the dataframe

    Returns
    -------
    String
        The text with words reduced to their respective lemmas based on their
        POS tag
    """

    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    text_with_lemma = " ".join(
        [lemmatizer.lemmatize(token, _get_pos_tag(token)) for token in tokens]
    )

    return text_with_lemma


def preprocess_text(text):
    text = text.replace("\n", " ")  # removing next line
    text = lower_casing(text)
    text = remove_urls(text)
    text = remove_html(text)
    # text = convert_emoji_to_words(text)
    # text = convert_emoticon_to_words(text)
    text = remove_stopwords(text)
    text = lemmatization_with_pos(text)
    text = remove_punctuation(text)
    # remove all characters except alphabets and numbers
    text = re.sub(r"[^a-zA-Z0-9 ]+", "", text)
    text = re.sub("( . )", " ", text)  # removing a single character word
    text = re.sub(r"\s+", " ", text)  # removing multiple white spaces
    text = text.strip()

    return text


if __name__ == "__main__":
    df = pd.read_csv("dataset/sample_dataset_real_or_fake_news.csv")
    df["preprocessed_text"] = df["text"].apply(lambda x: preprocess_text(x))
    df.drop(columns=['title', 'text', 'label'], inplace=True)
    df.to_csv("outputs/preprocessed-text-with-nltk.csv", index=False)
