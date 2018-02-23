'''
Embed sentences using gensim package. This gives a baseline to compare
my custom implementation. Gensim will, of course, be superior.
'''

from gensim.models import Word2Vec
from bokeh.plotting import figure, show
import pandas as pd

infile = 'sentences'

def read_sentences(infile):
    with open(infile) as f:
        lines = f.readlines()
    lines = [line.strip().split(',',) for line in lines]
    return lines

if __name__ == '__main__':
    sentences = read_sentences(infile)
    model = Word2Vec(sentences, sg = 0, size = 2)
    df = {k: model.wv[k] for k in model.wv.vocab}
    df = pd.DataFrame(df).T
    p = figure(title = "Gensim Word Embedding")
    p.scatter(x = df[0], y = df[1])
    p.text(x = df[0], y = df[1], text = df.index)
    show(p)
