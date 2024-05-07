# NLP using NLTK and spaCy

NLTK, or the Natural Language Toolkit, stands as a cornerstone in the realm of Natural Language Processing (NLP). NLTK offers a comprehensive suite of libraries and tools for tasks such as tokenization, lemmatization, stemming, tagging, parsing etc. It is easy to use and is used in various NLP applications like, sentiment analysis, machine translation, and information extraction for text cleaning and pre-processing steps.

Similar to NLTK, spaCy, too, represents a modern approach to NLP and is highly used for text preprocessing steps. Unlike NLTK, which has been a longstanding staple in the NLP community,spaCy emerged later, leveraging advancements in computational linguistics and machine learning. SpaCy is designed specifically for production use and helps you build applications that process and understand large volumes of text. It also provides pre-trained models that excel in various NLP tasks, including named entity recognition, part-of-speech tagging, dependency parsing, and text classification. It primarily focuses on performance optimization which has allowed it to process text swiftly, making it suitable for handling large-scale data.

One of the significant differences between SpaCy and NLTK lies in their design philosophies: while NLTK emphasizes flexibility and educational value, SpaCy prioritizes ease of use, performance, and modern NLP techniques. NLTK is more suited for educational purposes and prototyping, while spaCy is highly used to build production-ready NLP applications.


Let's discuss some of the text preprocessing steps used in NLP applications:

## 1) Text Normalization (Case Normalization)
Case normalization or lower casing simply deals with conversion of all input text into the same casing format (lower case) so that 'apple', 'Apple' and 'Apple' are all treated the same way. It ensures uniformity within the text.

This is more helpful for text featurization techniques like, tfidf as it helps to combine the same words together thereby reducing the duplication. However, this may not be helpful when we do tasks like Part of Speech tagging (where proper casing gives some information about nouns and so on) and Sentiment Analysis (where upper casing refers to anger and so on).

## 2) Removal of Punctuations
Punctuation marks such as commas, periods, quotation marks, and exclamation points serve grammatical functions in written language but often add noise and complexity to text data without contributing substantially to its semantic meaning. Thus eliminating punctuation during text preprocessing allows users to focus on extracting essential information and patterns from the text while reducing the dimensionality of the data. It allows us to treat "hello" and "hello!" the same way.

Various punctuation symbols can be extracted in python using ```string.punctuation```:

```!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~```

## 3) Removal of Stopwords
Stopwords are common words in a language that occur frequently but typically do not carry significant semantic meaning or contribute much to the understanding of a text. Examples of stopwords include articles (e.g., "the," "a," "an"), conjunctions (e.g., "and," "but," "or"), and prepositions (e.g., "in," "on," "at").

It is necessary to remove stopwords during NLP tasks for various reasons like:
- Noise Reduction: Stopwords add noise to the text data without providing valuable information. By removing them, NLP models can focus on extracting meaningful patterns and relationships from the text.
- Dimensionality Reduction: Stopwords occur frequently in text but do not contribute significantly to its semantic content. Removing stopwords reduces the dimensionality of the feature space, making it more manageable for analysis and reducing computational overhead.
- Improved Performance: Removing stopwords can lead to better performance in NLP tasks such as text classification and sentiment analysis. By focusing on content words rather than function words, NLP models can better capture the underlying meaning of the text.

## 4) Removal of Frequent Words
If we have a domain specific corpus, we might also have some frequent words which are of not so much importance to us and thus can be omitted to reduce computational burden and noise which ultimately leads to better performance.

## 5) Removal of Rare Words
Rare words having low frequency of occurrence in a corpus can introduce noise and hinder the performance of NLP models by adding unnecessary complexity and adding noise. By removing rare words, NLP systems can focus on learning more meaningful patterns and relationships from the text data, as rare words often lack sufficient contextual information for accurate analysis.

## 6) Removal of Emojis
Emojis, while expressive and widely used in informal communication, pose challenges for NLP models due to their non-standardized representation like, ```u"\U0001F600-\U0001F64F"```. Removing emojis helps to streamline the text data and remove noise from the text which can further assist in tokenization and feature extraction. However, since emojis too carry some contextual meaning of the text it would be better if we could convert the emojis into some meaningful word phrase which will be discussed in the subsequent sections.

## 7) Removal of Emoticons
Similar to emojis, emoticons too add noise to the text data and hinders in text tokenization. A basic difference between the two can be clarified as:
- :-) is an emoticon
- 😄 is an emoji

## 8) Conversion of Emoji to Words
Since emojis, too, carry some semantic information about the text, removing them completely can cause loss of information. This is mostly applicable for task like sentiment analysis where emojis express a lot about the sentiment of the given text. Hence, it is more suitable to convert them into words or phrases than completely eliminating from the text.

## 9) Conversion of Emoticons to Words
Similar to the above step, emoticons too can be converted into words or phrases to preserve the semantic meaning they posses. We will be referring to this [Github repo](https://github.com/NeelShah18/emot/blob/master/emot/emo_unicode.py) for the conversion of emojis and emoticons into words.

## 10) Removal of URLs
Real world data is noisy and contain unnecessary URLs like ```https://fusemachines.com/company/about/```. It is better to remove them completely from the text for NLP applications. URLs can be easily eliminated through pattern matching using regex. A probable regex pattern for a URL can be ```r'https?://\S+|www\.\S+'```

## 11) Removal of HTML tags
Another common preprocessing technique that will come handy in multiple places is removal of html tags like, ```<h1> Title </h1>```, ```<p> Some Text </p>```. When we scrape data from multiples websites we might also end up having html strings as part of the scraped text. HTML tags, too, can be removed using regex pattern matching as ```<.*?>```

## 12) Stemming
Stemming is a crucial preprocessing technique in natural language processing (NLP) aimed at reducing words to their root or base form, known as a stem. This process involves removing suffixes and prefixes from words to extract their core meaning, thereby reducing multiple variants of the same word and standardizing word representations.

For example, if there are three words in the corpus ```jumps```, ```jumping``` and ```jumped``` then stemming will stem the suffix to make them ```jump```.

One of the downside of stemming is that the result obtained after stemming i.e. the stem may or may not have meaning. For examples, the stem of words ```history``` and ```historical``` is ```histori``` which doesn't carry meaning.

Some examples of stemming are:
- root word ```go``` include:
    - goes
    - gone
    - going

- root word ```program``` include:
    - programming
    - programmer
    - programs

### Errors in Stemming
- **Over Stemming**
    - Over stemming occurs when a much larger part of a word is chopped off than what is required, which in turn leads to two or more words being reduced to the same root word.
    - For example, stemming algorithms may reduce the words ```university``` and ```universe``` to the same stem ```univers```, which would imply both the words have the same meaning which is completely false.

- **Under Stemming**
    - Under stemming occurs when two or more words which should have been reducted to the same stem get reduced to two different stems.
    - For example, ```data``` and ```datum``` might get reduced to ```dat``` and ```datu``` respectively when they both should have been reduced to ```dat```.

## 13) Lemmatization
Lemmatization stands as a very important preprocessing technique in NLP that transforms words into their canonical or base forms, known as lemmas. Unlike stemming, which simply chops off affixes to derive a word's root, lemmatization utilizes linguistic rules and morphological analysis to ensure that the resulting lemma is a valid word in the language.
For example lemmatization of the word ```feet``` is ```foot```.

Lemmatization works best when the word to be reduced is provided along with its POS tag. The POS tag of a word can be identified using ```pos_tag()``` method from nltk. So providing POS to the lemmatize method gives following results:
```
lemmatize("stripes", "v") --> strip
lemmatize("stripes", "n") --> stripe
```

## 14) Tokenization
Tokenization is a fundamental preprocessing step in NLP that involves breaking down a text into smaller units, typically words or subwords. These units are called tokens and serve as the building blocks for subsequent NLP tasks such as sentiment analysis, machine translation, and named entity recognition. The importance of tokenization lies in its ability to convert raw text into a format that machines can understand and process effectively. By segmenting text into tokens, NLP models can extract meaningful information, capture linguistic nuances, and perform analyses accurately.

- **Word Tokenization**
    - Sentences or texts are splitted in words.

- **Sentence Tokenization**
    - Texts are splitted in sentences on the basis of different punctuation symbols.

All the above mentioned preprocessing steps can be achieved through both **NLTK** and **spaCy** as illustrated in the python modules **nlp-using-nltk.py** and **nlp-using-spacy.py** respectively.
