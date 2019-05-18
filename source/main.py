import pandas
from sklearn.metrics import confusion_matrix, classification_report
import re
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
from keras.optimizers import RMSprop, SGD
from keras.utils import np_utils
from collections import Counter
import argparse
import time
import pickle


print("TF version: ", tf.__version__)
FILEPATH = "../data/amazon_reviews.csv"
RANDOM_SEED = 1
SAMPLE_FRAC = 0.1



def add_binary_label(df):
    """
    Converts the score attributes to a binary good/bad value. 1 and 2 is mapped to bad (0),
    4 and 5 is mapped to good (1) and 3 is removed
    :param df:
    :return:
    """
    new_array = []
    for e in df["Score"]:
        if e == 1 or e == 2:
            new_array.append(0)
        elif e == 3:
            new_array.append(-1)
        elif e == 4 or e == 5:
            new_array.append(1)
    new_df = df
    drop_indexes = [i for i, e in enumerate(new_array) if e == -1]
    new_df["label"] = new_array
    new_df = new_df.drop(drop_indexes)
    return new_df


def text_to_words(raw_tweet):
    """
    Only keeps ascii characters in the tweet and discards @words

    :param raw_tweet:
    :return:
    """
    letters_only = re.sub("[^a-zA-Z@]", " ", raw_tweet)
    words = letters_only.lower().split()
    meaningful_words = [w for w in words if not re.match("^[@]", w)]
    return " ".join(meaningful_words)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', help="Verbose output (enables Keras verbose output)", action='store_true', default=False)
    parser.add_argument('--gpu', help="Use LSTM/GRU gpu implementation", action='store_true', default=False)
    args = parser.parse_args()

    verbose = 1 if args.verbose else 2
    impl = 2 if args.gpu else 0

    start_time = time.time()
    print("Starting:", time.ctime())

    ############################################
    # Data


    # Load the data and select
    df = pandas.read_csv(FILEPATH)
    df = df[["Score", "Text"]]
    df = add_binary_label(df)
    df = df.sample(frac=SAMPLE_FRAC, random_state=RANDOM_SEED)

    # Make weighting adjustments for the solver to counter unbalanced data
    num_of_class_0 = len([i for i in df["label"] if i == 0])
    num_of_class_1 = len([i for i in df["label"] if i == 1])
    print(f"#Class 0: {num_of_class_0} \n#Class 1: {num_of_class_1}")
    scale = max(num_of_class_1, num_of_class_0)
    CLASS_WEIGHTS = {0 : scale/num_of_class_0, 1 : scale/num_of_class_1}

    # Pre-process the tweet and store in a separate column
    df['clean_text'] = df['Text'].apply(lambda x: text_to_words(x))



    # Join all the words in review to build a corpus
    all_text = ' '.join(df['clean_text'])
    words = all_text.split()

    # Convert words to integers
    counts = Counter(words)

    numwords = 5000  # Limit the number of words to use
    vocab = sorted(counts, key=counts.get, reverse=True)[:numwords]
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

    text_to_ints = []
    for each in df['clean_text']:
        text_to_ints.append([vocab_to_int[word] for word in each.split() if word in vocab_to_int])

    # Create a list of labels
    labels = np.array(df['label'])

    # Find the number of tweets with zero length after the data pre-processing
    text_len = Counter([len(x) for x in text_to_ints])
    print("Zero-length reviews: {}".format(text_len[0]))
    print("Maximum text length: {}".format(max(text_len)))


    # Remove those tweets with zero length and its corresponding label
    maximum_allowed_words = 500
    text_idx = [idx for idx, text in enumerate(text_to_ints) if len(text) > 0 and len(text) < maximum_allowed_words]
    labels = labels[text_idx]
    df = df.iloc[text_idx]
    text_to_ints = [text for text in text_to_ints if len(text) > 0 and len(text) < maximum_allowed_words]

    seq_len = min(max(text_len), maximum_allowed_words)
    features = np.zeros((len(text_to_ints), seq_len), dtype=int)
    for i, row in enumerate(text_to_ints):
        features[i, -len(row):] = np.array(row)[:seq_len]

    split_frac = 0.8
    split_idx = int(len(features) * 0.8)
    train_x, val_x = features[:split_idx], features[split_idx:]
    train_y, val_y = labels[:split_idx], labels[split_idx:]

    test_idx = int(len(val_x) * 0.5)
    val_x, test_x = val_x[:test_idx], val_x[test_idx:]
    val_y, test_y = val_y[:test_idx], val_y[test_idx:]

    print("\t\t\tFeature Shapes:")
    print("Train set: \t\t{}".format(train_x.shape),
          "\nValidation set: \t{}".format(val_x.shape),
          "\nTest set: \t\t{}".format(test_x.shape))

    print("Train set: \t\t{}".format(train_y.shape),
          "\nValidation set: \t{}".format(val_y.shape),
          "\nTest set: \t\t{}".format(test_y.shape))

    ############################################
    # Model
    drop = 0.0
    nlayers = 2  # >= 1
    RNN = LSTM  # GRU

    neurons = 128
    embedding = 40

    model = Sequential()
    model.add(Embedding(numwords + 1, embedding, input_length=seq_len, mask_zero=True))

    if nlayers == 1:
        model.add(RNN(neurons))
    else:
        model.add(RNN(neurons, return_sequences=True))
        for i in range(1, nlayers - 1):
            model.add(RNN(neurons, return_sequences=True))
        model.add(RNN(neurons))

    model.add(Dense(len(set(labels)), activation='softmax'))


    ############################################
    # Training

    learning_rate = 0.01
    optimizer = SGD(lr=learning_rate, momentum=0.95)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())

    epochs = 100
    batch_size = 256

    train_y_c = np_utils.to_categorical(train_y, len(set(labels)))
    val_y_c = np_utils.to_categorical(val_y, len(set(labels)))

    history = model.fit(train_x, train_y_c,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(val_x, val_y_c),
              verbose=verbose,
              class_weight=CLASS_WEIGHTS)
    file = open(f"""../output/history_{time.ctime().replace(" ", "_").replace(":", "~")}""", "wb")
    pickle.dump(history.history, file)
    file.close()

    model.save(f"""../output/model_{time.ctime().replace(" ", "_").replace(":", "~")}.h5""")


    ############################################
    # Results

    test_y_c = np_utils.to_categorical(test_y, len(set(labels)))
    score, acc = model.evaluate(test_x, test_y_c,
                                batch_size=batch_size,
                                verbose=verbose)
    print("Score:", score)
    print('Test ACC=', acc)

    test_pred = model.predict_classes(test_x, verbose=verbose)

    print()
    print('Confusion Matrix')
    print('-'*20)
    conf_matrix = confusion_matrix(test_y, test_pred)
    print(conf_matrix)
    print()
    print('Classification Report')
    print('-'*40)
    print(classification_report(test_y, test_pred))
    print()
    print("Ending:", time.ctime())
    print(f"Duration: {time.time()-start_time}")

