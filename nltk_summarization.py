import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import heapq

# Ensure the required NLTK data packages are downloaded
nltk.download('punkt')
nltk.download('stopwords')

def nltk_summarizer(raw_text):
    stop_words = set(stopwords.words("english"))
    word_frequencies = {}

    # Tokenizing the words and calculating word frequencies
    for word in word_tokenize(raw_text):
        word = word.lower()
        if word not in stop_words and word.isalpha():  # Ensure word is not a stopword and is alphabetic
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    # Finding the maximum frequency
    maximum_frequency = max(word_frequencies.values())

    # Normalizing the frequencies
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / maximum_frequency)

    # Tokenizing sentences
    sentence_list = sent_tokenize(raw_text)
    sentence_scores = {}

    # Scoring sentences based on word frequencies
    for sent in sentence_list:
        for word in word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:  # Prefer shorter sentences for summary
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    # Selecting the top 7 sentences
    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

    # Joining the sentences to form the summary
    summary = ' '.join(summary_sentences)
    return summary
