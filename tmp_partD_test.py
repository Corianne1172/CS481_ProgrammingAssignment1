import nltk
nltk.data.path.insert(0, '/Users/otiohkonan/nltk_data')

from nltk.corpus import brown, stopwords
from nltk import FreqDist
import math
import string

def tokenize(s: str):
    s = s.lower()
    tokens = []
    for w in s.split():
        w = w.strip(string.punctuation)
        if w != "":
            tokens.append(w)
    return tokens

def add_bounds(tokens):
    return ["<s>"] + tokens + ["</s>"]

def build_bigram_counts(tokens):
    unigrams = FreqDist(tokens)
    bigrams = FreqDist(nltk.bigrams(tokens))
    vocab = set(tokens)
    return unigrams, bigrams, vocab

def bigram_prob_mle(w1, w2, uni, bi):
    denom = uni[w1]
    if denom == 0:
        return 0.0
    return bi[(w1, w2)] / denom

def bigram_prob_laplace(w1, w2, uni, bi, V):
    return (bi[(w1, w2)] + 1) / (uni[w1] + V)

def sentence_details_and_ppl(sentence_tokens, uni, bi, vocab, mode):
    seq = add_bounds(sentence_tokens)
    bgs = list(nltk.bigrams(seq))

    probs = []
    log_sum = 0.0
    V = len(vocab) if len(vocab) > 0 else 1

    for (w1, w2) in bgs:
        if mode == "C":
            p = bigram_prob_mle(w1, w2, uni, bi)
        else:  # mode == "D"
            p = bigram_prob_laplace(w1, w2, uni, bi, V)

        probs.append(((w1, w2), p))

        if p == 0.0:
            return probs, float("inf")

        log_sum += math.log(p)

    N = len(bgs)
    ppl = math.exp(-log_sum / N)
    return probs, ppl

test_sentences = [
    "this is good",
    "this is bad",
    "i am good",
    "i am bad",
    "good afternoon",
    "good day",
]

stop_words = set(stopwords.words('english'))

# Model C: stopwords removed, unsmoothed bigram MLE
tokens_C = [w.lower() for w in brown.words() if w.isalpha() and w.lower() not in stop_words]
uni_C, bi_C, vocab_C = build_bigram_counts(tokens_C)

# Model D: stopwords kept, Laplace-smoothed bigram
tokens_D = [w.lower() for w in brown.words() if w.isalpha()]
uni_D, bi_D, vocab_D = build_bigram_counts(tokens_D)

print("Perplexity comparison (lower is better)\n")

for s in test_sentences:
    sent_toks = [t for t in tokenize(s) if t.isalpha()]

    probs_C, ppl_C = sentence_details_and_ppl(sent_toks, uni_C, bi_C, vocab_C, mode="C")
    probs_D, ppl_D = sentence_details_and_ppl(sent_toks, uni_D, bi_D, vocab_D, mode="D")

    print("Sentence:", s)
    print("Model C perplexity:", ppl_C)
    for bg, p in probs_C:
        print("  ", bg, "->", p)

    print("Model D perplexity:", ppl_D)
    for bg, p in probs_D:
        print("  ", bg, "->", p)

    print()