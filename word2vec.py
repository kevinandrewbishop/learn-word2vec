'''
Manual implementation of word2vec (i.e. only uses numpy. No NLP packages).

The purpose is to learn exactly how word2vec embeds words in vector space.
Therefore the code here is not concerned with efficiency or having all the
bells and whistles. It deliberately is bare bones to make the logic of
word2vec clear without any clutter of housekeeping.

'''
import numpy as np

def read_sentences(infile):
    '''
    Helper function to read in a file of sentences.
    File should have the following format:

        word1,word2,word3...wordN
        word1,word2,word3...wordN

    Each row is a sentence. Each word is separated by commas.
    '''
    with open(infile) as f:
        lines = f.readlines()
    lines = [line.strip().split(',',) for line in lines]
    return lines


def build_vocab(sentences, dim = 2):
    '''
    Assigns each word a unique integer ID. Creates a context vector and
    target vector for each word. Stored as two 2d numpy arrays.
    '''
    vocab = dict()
    i = 0
    for sentence in sentences:
        for word in sentence:
            if word in vocab:
                continue
            vocab[word] = i
            i += 1
    #randomly initialize context vectors
    con_vec = np.random.rand(i,dim) - .5
    #randomly initialize target vectors
    tar_vec = np.random.rand(i,dim) - .5
    return vocab, con_vec, tar_vec

def train(sentences, con_vec, tar_vec, vocab, window = 5, neg_samples = 5, alpha = .05):
    '''
    The main function to train the vectors.


    '''
    #TODO: currently this changes the vectors in place -not great.
    #Ideally all these functions should be methods in a class.
    for sentence in sentences:
        for i, word in enumerate(sentence):
            #calculate start and stop points for the context window
            start = max(0, i - window)
            stop = min(len(sentence), i + window + 1)
            context_words = sentence[start:i] + sentence[i+1:stop]
            train_pair(context_words, word, con_vec, tar_vec, vocab)


def train_pair(context_words, target_word, con_vec, tar_vec, vocab, neg_samples = 5, alpha = .05):
    '''
    A single iteration of training on context words and target word. This
    is the meat of word2vec.
    '''
    #Extract context word indices and vectors
    con_words_ind = [vocab[w] for w in context_words]
    con_words_vec = con_vec[con_words_ind]
    con_words_vec_mean = con_words_vec.mean(axis = 0)

    #Extract target word indices and vectors
    tar_word_ind = vocab[target_word]
    tar_words_ind = get_negative_samples(tar_word_ind, neg_samples, len(vocab))
    tar_words_vec = tar_vec[tar_words_ind]

    con_tar_dot = np.dot(con_words_vec_mean, tar_words_vec.T)
    activation = logistic(con_tar_dot)
    labels = np.zeros(neg_samples)
    labels[0] = 1
    error = (labels - activation)*alpha #alpha learning rate
    #vector we'll use to update target vector embedding
    tar_vec_update = np.outer(error, con_words_vec_mean)
    tar_vec[tar_words_ind] += tar_vec_update

    #vector we'll use to update context vector embedding
    con_vec_update = np.dot(error, tar_words_vec)
    for ind in con_words_ind:
        con_vec[ind] += con_vec_update

def logistic(x):
    '''
    Logistic function.
    '''
    return 1/(1+np.exp(-x))

def get_negative_samples(target_word_ind, neg_samples, vocab_size):
    '''
    Helper function to randomly select "negative" examples for training.
    Note this is really stripped down. Usually words are selected weighted
    by their frequency in the overall corpus. Here  I just uniformly sample.
    '''
    tar_words_ind = [target_word_ind]
    #keep randomly sampling until you find enough negative samples
    while len(tar_words_ind) < neg_samples:
        ind = np.random.randint(vocab_size)
        if ind != target_word_ind:
            tar_words_ind.append(ind)
    return tar_words_ind




if __name__ == '__main__':
    #name of file containing "sentences"
    infile = 'sentences'

    #Train the vectors
    sentences = read_sentences(infile)
    vocab, con_vec, tar_vec = build_vocab(sentences)
    train(sentences, con_vec, tar_vec, vocab)

    #Do some plots to visually inspect
    from bokeh.plotting import figure, show
    import pandas as pd
    inverse_vocab = {v: k for k, v in vocab.items()} #maps from index to word
    df = pd.DataFrame(con_vec)
    df.index = [inverse_vocab[i] for i in df.index] #get labels
    p = figure(title = "Word Embedding")
    p.scatter(df[0], df[1])
    p.text(df[0], df[1], text = df.index)
    show(p) #render in the browser


'''
Here's what I've gathered from plowing through the gensim source code (as
well as reading plenty of tutorials).

CBOW
Prep:
    Assign each word to an index number.
    Assign each word a randomly initialized context vector (syn0)
    Assign each word a randomly initialized target vector (syn1neg)

Train:
    Given a sentence loop over each word in the sentence and
        Randomly select a subset of words within the context window
        Extract the context vectors of each word in that window (found in syn0)
        Compute mean of those vectors, save as "l1"

        Look up the target vector of the target word
        Randomly select some "negative" target words (e.g. 5 of them)
        [details of how to sample are ignored]
        Look up the target vectors of those negative target words (found in syn1neg!!!)
        Save the target word and negative words in a matrix called l2b
            Be sure the true target word is the first row in the matrix, and
            all negative context words are in subsequent rows
        Compute dot product of l1 (mean context vector) with each of the rows
            in l2b. Result is dot product with true target word, and with
            each negative example. Save result in "prod_term"
        Apply logistic function to each element of prod_term, save as "fb"
        Compute error which is [1, 0, 0... N] - fb where N is number of
            negative samples
        Multiply error by learning rate alpha, save as "gb"

        Update target vector (syn1neg)
            multiply l1 by each element of gb for example:
                gb[0] is error rate of the true target word.
                Multiply l1 by gb[0].
                If gb[0] if positive (meaning yhat was too small) then the result
                will be a tiny version of l1 (scaled because error is small and alpha
                is small).
                if gb[0] is negative (meaning yhat was too big) then the result
                will be a tiny negative version of l1.
                Add the result to the target vector. This makes the tiny gradient
                update to make the target vector slightly more or slightly less
                like the mean context vector l1, depending on whether the error
                was positive or negative.
        Update context vector (syn0)
            Compute dot product of gb and l2b. This ends up being an "udpate vector"
            where each element is a weighted sum of the error on the ith target word
            and the jth value of the ith target word. For example:
                If there are 3 target words (true match, false match A, false match B)
                and each is embedded in 2 dimensional space, and the error (gb)
                is [.1, -.2, .03] the value at update vector[0] will be
                true_match[0]*.1 + false_matchA[0]*-.2 + false_matchB[0]*.03
                and update vector[1] will be
                true_match[1]*.1 + false_matchA[1]*-.2 + false_matchB[1]*.03
            For each context word vector
                add the update vector to it
                (since we computed the mean of the context vectors, we can't
                break them out individually to see how much each contributed to
                the error, so we must update them all using the same amount)
'''
