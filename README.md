# learn-word2vec

Manually implementing word2vec in order to learn it.  

## What is Word2Vec
Word2Vec is a method for transforming words like "dog" or "hello," which computers aren't very good at analyzing, into
vectors of numbers like [0.23, 1.44, -.2] and [0.1, 0.223, .11] which computers *are* good at analyzing. Once we have these
"vector representations" of words, we can run them through all sorts of traditional algorithms as well as
interesting newer techniques like deep neural networks.  

For more background on why we'd want to do this and high level views for how it's done, see 
[tensorflow's tutorial](https://www.tensorflow.org/tutorials/word2vec) 
as well as the excellent [Colah's blog](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/).  

## Why did I make this
I recently went to a [natural language processing conference](http://acl2017.org/). Before I arrived, I understood how 
basic shallow and deep neural nets worked, but I knew next to nothing of this "word embedding" people kept referring to.
I quickly learned that "word embedding" is step 1 in the vast majority of cutting edge NLP methods. So I read a few
blogs to get a basic idea of what they were before returning to the conference.  

This gave me enough of an understanding to competently use a package like [gensim](https://radimrehurek.com/gensim/)
or explain it at a high level to someone. But I didn't *really* understand exactly what was happening under the hood.  

Blogs I read were either too high level or, when they did dig into the details, made too much use of (sorry) clumsy APIs 
like tensorflow. Academic papers used math notation rather than code or pseudocode, which I'm rusty on. 

The only way I really learn an algorithm is to write it myself. At least a basic version. So I dug into gensim's source
code to understand what was actually happening. Once I was confident I understood it, I wrote my own version without
any bells or whistles to obfuscate the basic mechanics of word2vec. If I start up an analytics blog, this will
definitely be an entry.

## How did I make this
Word2Vec is trained on a corpus of text. It could be a ton of google news articles, wikipedia pages, novels, etc. I wanted
an extremely simple "corpus" that I could control to use as a test. So I generated a set of fake sentences via a method
loosely inspired by [topic modeling](https://en.wikipedia.org/wiki/Topic_model). I have a set of topics like "breakfast"
"animals" "work" "cute things" etc. I associate with each topic a set of 5-10 words like "eggs" or "computer." Some words
fall in multiple topics (e.g. "dogs" fall under "animals" as well as "cute things"). I then randomly generate thousands of
"sentences" about each topic (lists of words). So a breakfast sentence might look like "eggs bacon coffee coffee early."
I also occasionally inject random off-topic words for (at least a hint of) realism.  

I used the sentences as training data on gensim's Word2Vec model as a benchmark and proof of concept. As expected, gensim
does a great job clustering all the various topics together. Words that are part of multiple topics tend to appear in the
space between the clusters. Concretely, when I plot each word's 2d vector, words that share a topic show up in the same
area.  

Since I knew it was possible with gensim, it should work if I coded it up properly. Sure enough, the vectors by and large
made sense. It wasn't quite as neat as genism's results, but it was clearly doing the right thing.
