# Importing libraries
import nltk
nltk.data.path.insert(0, '/Users/otiohkonan/nltk_data')
import math
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk import FreqDist
import matplotlib.pyplot as plt

# Defining the two corpora
brownCorpus = nltk.corpus.brown
reutersCorpus = nltk.corpus.reuters

# Defining stop words
stop_words = set(stopwords.words('english'))

# Extracting words
brownWords = brownCorpus.words()
reutersWords = reutersCorpus.words()

# Lowercasing + removing stopwords
filtered_brownWords = [
    word.lower() for word in brownWords if word.lower() not in stop_words
]

filtered_reutersWords = [
    word.lower() for word in reutersWords if word.lower() not in stop_words
]

# Frequency distributions
brown_freq = FreqDist(filtered_brownWords)
reuters_freq = FreqDist(filtered_reutersWords)

# Top 10 words
top10_brown = brown_freq.most_common(10)
top10_reuters = reuters_freq.most_common(10)

# Displaying results
print("Top 10 words in Brown Corpus:")
for rank, (word, freq) in enumerate(top10_brown, start=1):
    print(f"{rank}. {word} ({freq})")

print("\nTop 10 words in Reuters Corpus:")
for rank, (word, freq) in enumerate(top10_reuters, start=1):
    print(f"{rank}. {word} ({freq})")
    
# Get top 1000 words
top1000_brown = brown_freq.most_common(1000)
top1000_reuters = reuters_freq.most_common(1000)

# Prepare log(rank) and log(frequency) for Brown
brown_ranks = [math.log(rank) for rank in range(1, 1001)]
brown_freqs = [math.log(freq) for (_, freq) in top1000_brown]

# Prepare log(rank) and log(frequency) for Reuters
reuters_ranks = [math.log(rank) for rank in range(1, 1001)]
reuters_freqs = [math.log(freq) for (_, freq) in top1000_reuters]

# Plot Brown
plt.figure()
plt.plot(brown_ranks, brown_freqs)
plt.xlabel("log(rank)")
plt.ylabel("log(frequency)")
plt.title("Brown Corpus: log(rank) vs log(frequency)")
plt.show()

# Plot Reuters
plt.figure()
plt.plot(reuters_ranks, reuters_freqs)
plt.xlabel("log(rank)")
plt.ylabel("log(frequency)")
plt.title("Reuters Corpus: log(rank) vs log(frequency)")
plt.show()

# Words to analyze
technical_word = "algorithm"
casual_word = "food"

# Total word counts
total_brown_words = len(filtered_brownWords)
total_reuters_words = len(filtered_reutersWords)

# Frequency counts
brown_tech_count = brown_freq[technical_word]
brown_casual_count = brown_freq[casual_word]

reuters_tech_count = reuters_freq[technical_word]
reuters_casual_count = reuters_freq[casual_word]

# Unigram probabilities
brown_tech_prob = brown_tech_count / total_brown_words
brown_casual_prob = brown_casual_count / total_brown_words

reuters_tech_prob = reuters_tech_count / total_reuters_words
reuters_casual_prob = reuters_casual_count / total_reuters_words

# Display results
print("\nUnigram Probabilities\n")

print("Technical word:", technical_word)
print("Brown -> count:", brown_tech_count,
      "probability:", brown_tech_prob)
print("Reuters -> count:", reuters_tech_count,
      "probability:", reuters_tech_prob)

print("\nNon-technical word:", casual_word)
print("Brown -> count:", brown_casual_count,
      "probability:", brown_casual_prob)
print("Reuters -> count:", reuters_casual_count,
      "probability:", reuters_casual_prob)