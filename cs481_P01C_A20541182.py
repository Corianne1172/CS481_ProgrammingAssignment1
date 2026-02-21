# Importing libraries
import nltk
nltk.data.path.insert(0, '/Users/otiohkonan/nltk_data')

from nltk.corpus import brown, stopwords
from nltk import FreqDist

# --- Build corpus tokens: lowercase + remove stopwords + keep alphabetic only ---
stop_words = set(stopwords.words('english'))

brown_tokens = [w.lower() for w in brown.words() if w.isalpha() and w.lower() not in stop_words]

# Unigram and bigram frequency distributions
unigram_freq = FreqDist(brown_tokens)
bigram_freq = FreqDist(nltk.bigrams(brown_tokens))

def top3_followers(w1: str):
    """
    Returns a list of up to 3 tuples: (w2, P(w2|w1))
    P(w2|w1) = count(w1,w2) / count(w1)
    """
    denom = unigram_freq[w1]
    if denom == 0:
        return []

    # Collect all w2 that follow w1
    followers = {}
    for (a, b), c in bigram_freq.items():
        if a == w1:
            followers[b] = c / denom

    # Sort by probability desc, then word asc (stable tie-break)
    sorted_followers = sorted(followers.items(), key=lambda x: (-x[1], x[0]))
    return sorted_followers[:3]

# --- 1) Ask user for initial word W1, lowercase, validate ---
sentence = []

while True:
    W1 = input("Enter initial word/token W1: ").strip().lower()

    if W1 in unigram_freq:
        sentence.append(W1)
        break

    print(f"'{W1}' is not in the (stopword-removed) Brown corpus.")
    print("1) Ask again")
    print("2) QUIT")
    choice = input("Choose 1 or 2: ").strip()

    if choice == "2":
        print("QUIT")
        raise SystemExit
    
# --- 2) Repeatedly offer top-3 next words until user quits ---
while True:
    current = sentence[-1]
    options = top3_followers(current)

    print(f"\n{current} ...")
    print("Which word should follow:")

    # If fewer than 3 options exist, show whatever exists
    for i, (w2, prob) in enumerate(options, start=1):
        print(f"{i}) {w2}  P({current} {w2}) = {prob}")

    print("4) QUIT")

    user_choice = input("Enter 1, 2, 3, or 4: ").strip()

    if user_choice == "4":
        break

    # If invalid (not 1/2/3/4), assume 1
    if user_choice not in {"1", "2", "3"}:
        user_choice = "1"

    idx = int(user_choice) - 1

    # If user picks 2/3 but we have fewer options, default to 1
    if idx >= len(options):
        idx = 0

    next_word = options[idx][0]
    sentence.append(next_word)

print("\nFinal generated sentence:")
print(" ".join(sentence))
