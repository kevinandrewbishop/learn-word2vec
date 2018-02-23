'''
Creates a fake set of "sentences" sort of like topic modeling sentence
generation model. Select a few "topics" and a set of words or "vocabulary."
Each word belongs to one or more topics.

In the example below, certain words belong to one topic (e.g. "eggs" belongs in "breakfast").
Other words fall under multiple topics (e.g. "cofee" falls under "breakfast" and "work").

'''
from random import choice, random, randint


outfile = 'sentences'

topics =['breakfast', 'dinner', 'animal', 'cute', 'work']

words = [
    ['coffee', 'eggs', 'sausage', 'early', 'morning', 'pancake'],
    ['steak', 'chicken', 'wine', 'late', 'evening'],
    ['dog', 'cat', 'chicken', 'rabbit', 'horse', 'monkey'],
    ['baby', 'dog', 'rabbit', 'cat', 'button', 'toy'],
    ['suit', 'coffee', 'computer', 'report', 'early', 'morning']
]


def make_sentences(topics, words, N = 10000):
    sentences = list()
    for i in range(N):
        #ind is the index of the topic
        ind = randint(0, len(topics)-1)
        sentence = [] #start a blank sentence
        for i in range(5):
            flip = random() < 0.2 #randomly change topic to add variation
            topic = topics[ind - flip]
            w = choice(words[ind])
            sentence.append(w)
        sentences.append(sentence)
    return sentences



if __name__ == '__main__':
    N = 10000
    sentences = make_sentences(topics, words, N)
    sentences = '\n'.join([','.join(sentence) for sentence in sentences])
    with open(outfile, 'w') as f:
        f.write(sentences)
