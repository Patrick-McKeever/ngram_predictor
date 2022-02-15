import os
import sys

import nltk
nltk.download('brown')
from nltk.corpus import brown

# Turn off useless tensorflow warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras.utils.np_utils
import numpy as np
from keras import layers
from keras.preprocessing.sequence import pad_sequences
import keras.preprocessing.text as kpp

# Takes a second to initialize, better to just do it once.
TOKENIZER = kpp.Tokenizer()

'''
Take a list of sentences, turn them into a list of ngrams. For each sentence,
append to the list a list of words 1..n for all values of n less than the
length of the sentence. E.g. "Je vous ai compris" -> [["Je"], ["Je", "vous"],
["Je", "vous", "ai"], ["Je", "vous", "ai", "compris"].
If this structure seems odd, I'm using it because it's what bidirectional LSTMs
take as input: 
towardsdatascience.com/nlp-text-generation-through-bidirectional-lstm-model-9af29da4e520

@input sentences List of sentences
@output (The aforementioned list, the length of the longest ngram in the list).
'''
def ParseText(sentences):
    TOKENIZER.fit_on_texts(sentences)

    # We'll need to know max ngram len for padding.
    max_ngram_len = 0
    corpus_ngrams = []
    for sent in sentences:
        # Make each sentence into a sequence of numerical "tokens", with each token
        # corresponding to a word in the sentence (excluding punctuation and junk).
        toks = TOKENIZER.texts_to_sequences([sent])[0]
        # The ngrams here are going to differ slightly from the style we used in class.
        # Since we want to train an NN to generate text one word at a time based on the
        # previous words in a sentence, we should chop the sentence into all sequences of
        # words [0,n] for n = 1..(len of sentence). I.e. ngrams[0] will be toks[:2],
        # ngrams[1] will be toks[:3], etc.
        corpus_ngrams += [toks[:i + 1] for i in range(1, len(toks))]
        max_ngram_len = max(max_ngram_len, len(corpus_ngrams[-1]))

    return corpus_ngrams, max_ngram_len

'''
Put a truncated version of the brown corpus through the ParseText func.
@output (ngram list from brown, max ngram length of brown).
'''
def ParseBrown():
    plaintext_corpus = ' '.join(brown.words()[:10000]).lower()
    sentences = nltk.sent_tokenize(plaintext_corpus)
    return ParseText(sentences)

'''
Train a neural network to predict the next word of each sentence based
on each previous word.
'''
def Train():
    corpus_ngrams, max_ngram_len = ParseBrown()

    # Neural networks work with matrices. For them to function, they need to accept
    # matrices of fixed dimensions (bit of an oversimplification, but whatever).
    # Thus, for every row, we need to prepend a bunch of 0s to ensure that those
    # rows are the same length.
    ngram_matrix = np.array(pad_sequences(corpus_ngrams, maxlen=max_ngram_len,
                                          padding='pre'))
    # All rows, up exclude last column.
    indep_vars = ngram_matrix[:, :-1]
    # Only last column.
    seq_labels = ngram_matrix[:, -1]
    # This function is a bit complex. Essentially, it takes a vector and transforms it
    # into a matrix with a number of rows equivalent to the length of the vector and a
    # number of columns equal to the value of positional element num_classes. This allows
    # us to split some categorical (i.e. discrete) dependent variable into numerical a
    # number of (usually T/F) numerical values in a process called one-hot encoding
    # (see: https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f).
    dep_columns = len(TOKENIZER.word_index) + 1
    # Essentially, we want the neural network to predict the final form of the
    # sentence based on each previous form of the sentence.
    dep_vars = keras.utils.np_utils.to_categorical(seq_labels,
                                                   num_classes=dep_columns)

    # This is a modified version of Namrata Kapoor's architecture
    # towardsdatascience.com/nlp-text-generation-through-bidirectional-lstm-model-9af29da4e520
    # I remove the dropout and LSTM layers after the bidirectional layer because
    # they were taking too long to train.
    model = keras.Sequential()
    model.add(layers.Embedding(dep_columns, 64, input_length=max_ngram_len - 1))
    model.add(layers.Bidirectional(layers.LSTM(20)))
    model.add(layers.Dense(dep_columns, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(indep_vars, dep_vars, epochs=500, verbose=1)

    model.save('./model.tf', save_format='tf')

'''
Use the model to finish a sentence.
@input seed The start of a sentence to finish.
@input model A sequential model which can finish a given sentence.
@input num_words The number of words to append to the existing sentence.
@output The completed sentence.
'''
def GenFromPrompt(seed, model, num_words):
    corpus_ngrams, max_ngram_len = ParseBrown()

    for _ in range(num_words):
        token_list = TOKENIZER.texts_to_sequences([seed])[0]
        token_list = pad_sequences([token_list], maxlen=max_ngram_len - 1, padding='pre')
        predicted_x = model.predict(token_list, verbose=0)
        predicted = np.argmax(predicted_x, axis=1)
        output_word = ""
        for word, index in TOKENIZER.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed += " " + output_word
    return seed

'''
If user runs as "python3 main.py train", train the model.
Otherwise, let them suggest prompts for the model and get results.
'''
def Main():
    # TODO: use argparse for this.
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        Train()

    if not os.path.exists('model.tf'):
        sys.exit('\nTrained model does not exist in this directory ("model.tf" not found).\n' 
                 'To train model, run "python3 main.py train".')

    model = keras.models.load_model('model.tf')
    while True:
        prompt = input('Enter prompt (q to quit):\t')
        try:
            num_words = int(input('Enter num words to generate:\t'))
        except ValueError:
            num_words = 5

        if prompt == 'q':
            break
        print(f'Model-generated text: {GenFromPrompt(prompt, model, num_words)}')


if __name__ == '__main__':
    Main()