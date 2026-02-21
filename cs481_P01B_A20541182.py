# Importing libraries
import nltk
import math
import string
nltk.data.path.insert(0, '/Users/otiohkonan/nltk_data')
from nltk.corpus import brown
from nltk import FreqDist

#download the brown corpus
brownCorpus = nltk.corpus.brown

#ask the user for a sentence
S = input("Please enter a sentence: ")

#lowercase the sentence
S = S.lower()

# Tokenize (simple split) + OPTIONAL punctuation stripping for cleaner tokens
# If you want punctuation kept, remove the strip() part.
tokens = [w.strip(string.punctuation) for w in S.split() if w.strip(string.punctuation) != ""]

# Add start/stop tokens
words = ["<s>"] + tokens + ["</s>"]

#get the words in the brown corpus and lowercase them
brown_words = [w.lower() for w in brown.words()]

#create bigrams from the brown corpus
brown_bigrams = list(nltk.bigrams(brown_words))

#create frequency distributions for unigrams and bigrams from the brown corpus
unigram_freq = FreqDist(brown_words)
bigram_freq = FreqDist(brown_bigrams)

#create bigrams from the input sentence
sentence_bigrams = list(nltk.bigrams(words))

#calculate the probability of the sentence using the bigram model
P_S = 1.0
bigram_probs = []

for (w1, w2) in sentence_bigrams:
    # Boundary bigrams: <s> -> first_word OR last_word -> </s>
    if w1 == "<s>" or w2 == "</s>":
        prob = 0.25
    else:
        denom = unigram_freq[w1]
        if denom == 0:
            prob = 0.0
        else:
            prob = bigram_freq[(w1, w2)] / denom

    bigram_probs.append(((w1, w2), prob))
    P_S *= prob
    
# Display results
print("\nSentence lowercased:")
print(S)

print("\nBigrams and probabilities:")
for bigram, prob in bigram_probs:
    print(f"{bigram} -> {prob}")

print("\nFinal sentence probability P(S):")
print(P_S)