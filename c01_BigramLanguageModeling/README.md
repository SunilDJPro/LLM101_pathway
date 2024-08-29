#Bigram Language Modeling

Bigram language modeling is a technique used in natural language 
processing (NLP) to estimate the probability of a word given its preceding 
words. It is a type of statistical language model that assumes the 
probability distribution of a word depends on the previous word or bigram 
(two consecutive words). The key idea behind bigram models is that words 
tend to appear together frequently in a corpus of text, so if we have seen 
a certain sequence of words before, we can use this information to predict 
the next word more accurately than if we only considered each word 
independently.

A bigram model estimates the probability of a word given its preceding 
word: P(w|w-1), where w is the current word and w-1 is the previous word. 
This probability can be calculated using maximum likelihood estimation 
(MLE) or smoothing techniques like additive smoothing to handle rare words 
or zero counts.

To build a bigram language model, we need a corpus of text data that we 
can use to estimate the frequency of bigrams. We can preprocess this data 
by tokenizing it into words and converting all letters to lowercase for 
case-insensitivity. Then, we can count the number of occurrences of each 
bigram in the corpus and compute their probabilities using MLE or 
smoothing techniques. Once we have estimated the probabilities, we can 
use them to predict the next word given a sequence of words.

Basic use cases of Bigram Modeling are:

    1) Spell checking: Given an input word, a bigram model can help suggest 
    corrections based on the most probable continuation.

    2) Text generation: You can generate random text by sampling the most 
    likely next word given a sequence of words using the estimated 
    probabilities.

    3) Part-of-speech tagging: Bigrams can be used to predict the 
    part-of-speech (POS) tag for the next word based on the current POS tag 
    and the previous word.

    4) Sentence segmentation: You can use bigrams to determine where sentences 
    end in a text by finding the most probable sentence boundary given the 
    preceding words.

    5) Named entity recognition: Bigrams can help identify named entities 
    in text by finding sequences of words that are more likely to occur 
    together, such as person names or organization names. For example, the 
    bigram model might learn that the sequence "New York" is more likely to be 
    a location than a person name.

    6) Machine translation: Bigrams can be used in machine translation to 
    estimate the probability of a word in the target language given its 
    corresponding word or phrase in the source language. For example, if you 
    see the source words "il faut", the bigram model can help predict the most 
    probable French translation for the English word "it is necessary".

#Tri-gram Modeling 

A trigram language model is an extension of the bigram model that 
considers three consecutive words instead of two. It estimates the 
probability of a word given its preceding two words: P(w|w-1, w-2). The 
key idea behind trigram models is to capture higher-order dependencies in 
text beyond just bigrams.

Trigram models can be more accurate than bigram models and provide better 
predictions for some tasks. However, they require more data and 
computational resources to estimate and may not always improve performance 
significantly compared to n-gram models with larger n. N-gram models that 
consider even more previous words (such as 4-grams or 5-grams) can capture 
even higher-order dependencies in text, but they become increasingly 
sparse and computationally expensive to estimate.

An example for the tri-gram modelling:

    Let's say we have the following text corpus: "The quick brown fox 
    jumped over the lazy dog. The dog slept under the tree."

    To build a trigram model for sentence segmentation, we can first 
    preprocess the text by tokenizing it into words and converting all letters 
    to lowercase: ["the", "quick", "brown", "fox", "jumped", "over", "the", 
    "lazy", "dog", ".", "the", "dog", "slept", "under", "the", "tree", "."]

    Next, we can count the number of occurrences of each trigram in the corpus 
    and compute their probabilities using MLE or smoothing techniques. For 
    example, the trigram ". t" (period followed by a space and the word "t") 
    has a high probability because it occurs frequently at the beginning of 
    sentences.

    If we see the input sequence "the lazy dog", we can use the trigram 
    model to predict the most probable sentence boundary given this sequence. 
    The bigram "lazy dog" is unlikely to end a sentence based on our training 
    data, but the trigram "dog ." (followed by a period) has high probability, 
    so we can infer that the next word is likely to be the beginning of a new 
    sentence.

#n-gram modeling

An n-gram language model is a generalization of the bigram or trigram 
model that considers n consecutive words instead of just two or three. The 
key idea behind n-gram models is to capture higher-order dependencies in 
text beyond bigrams or trigrams. For example, an n-gram model with n=4 
(quadrigram) considers four consecutive words: P(w|w-1, w-2, w-3).

As n increases, n-gram models capture even higher-order dependencies in 
text and provide more accurate predictions for some tasks. However, they 
become increasingly sparse and computationally expensive to estimate. 
Additionally, higher-order n-grams may not always improve performance 
significantly compared to lower-order n-grams or other models (e.g., 
neural language models).

Smoothing is a technique used in n-gram models to address the sparsity 
problem, where some n-grams may not appear in the training data and have 
zero probability. This can occur when the size of the vocabulary is large 
or when the corpus is small. Smoothing techniques help assign non-zero 
probabilities to these n-grams based on the probabilities of their 
constituent sub-n-grams (e.g., bigrams, trigrams) and other heuristics.

Few smoothing techniques for n-gram modelling:

    1. Additive smoothing: This technique adds a fixed value (usually 1 or a 
    small constant) to the count of each n-gram in the training data before 
    computing their probabilities. For example, if we have an n-gram "w" that 
    appears m times in the corpus and another n-gram "v w" that appears n 
    times, we can compute the smoothed probability of "w" as:
    P_smooth(w) = (m + 1) / (|V|^n + N), where |V| is the size of the 
    vocabulary and N is the total number of n-grams in the corpus.

    2. Laplace smoothing: This technique adds a small constant (usually 1 or a 
    small value) to each count in the training data before computing 
    probabilities. For example, if we have an n-gram "w" that appears m times 
    and another n-gram "v w" that appears n times, we can compute the smoothed 
    probability of "w" as:
    P_smooth(w) = (m + 1) / (n + |V|), where |V| is the size of the 
    vocabulary.

    3. Kneser-Ney smoothing: This technique models the frequency distribution 
    of n-grams based on their frequency in the training data and their 
    frequency as the middle word in a trigram. It assigns non-zero 
    probabilities to rare n-grams that occur frequently as the last word in a 
    trigram.

    4. Interpolation smoothing: This technique combines additive or Laplace 
    smoothing with backoff estimates for lower-order n-grams. For example, if 
    an n-gram "v w x" has zero probability based on the training data, we can 
    use a lower-order n-gram "w x" as a fallback estimate.

Smoothing techniques help improve the accuracy and efficiency of n-gram 
models by assigning non-zero probabilities to all n-grams, even those that 
do not appear in the training data. However, they may not always provide 
significant improvements compared to other modeling techniques or 
architectures (e.g., neural language models).




