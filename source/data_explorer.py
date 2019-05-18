import numpy as np
import pandas
import re
import matplotlib.pyplot as plt

FILEPATH = "../data/amazon_reviews.csv"

def text_to_words(raw_tweet):
    """
    Only keeps ascii characters in the tweet and discards @words

    :param raw_tweet:
    :return:
    """
    letters_only = re.sub("[^a-zA-Z@]", " ", raw_tweet)
    words = letters_only.lower().split()
    meaningful_words = [w for w in words if not re.match("^[@]", w)]
    return meaningful_words

def hist_word_count(df):
    word_counts = []
    for text in df["Text"]:
        meaningful_words = text_to_words(text)
        word_counts.append(len(meaningful_words))
    plt.hist(word_counts,  bins=100)
    plt.show()

def hist_label_frequencies(counting_dict):
    cd = counting_dict
    weights = [cd[1], cd[2], cd[3], cd[4], cd[5]]
    values = list(range(1, 6))
    plt.hist(values, bins=5, weights=weights)
    plt.show()


def label_frequencies(df):
    counting_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for e in df["Score"]:
        counting_dict[int(e)] += 1
    print("Score frequencies: ", counting_dict)
    return counting_dict


if __name__ == '__main__':
    df = pandas.read_csv(FILEPATH)
    counting_dict = label_frequencies(df)
    hist_label_frequencies(counting_dict)
