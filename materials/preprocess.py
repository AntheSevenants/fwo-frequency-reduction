import argparse
import os
import random
import pandas as pd

from gensim.models import KeyedVectors

parser = argparse.ArgumentParser(description='preprocess - sample words from a CGN frequency list and associate them with vectors')
parser.add_argument('frequency_path', type=str, help='the file path to your COREX frequency list')
parser.add_argument('vectors_path', type=str, help='the file path to your vector file')
parser.add_argument('sample_n', type=int, help='the number of words to sample')
parser.add_argument('output_path', type=str, help='the output path for the vectors')
args = parser.parse_args()

N_CUTOFF = 10000

if args.sample_n < N_CUTOFF:
    raise ValueError(f"Sample size is bigger than cutoff size ({N_CUTOFF})")

print("Reading frequency list")
df = pd.read_csv(args.frequency_path, sep="\t")
print("Removing tagged entries")
df = df[df['tag'].isna()]
print(f"Limiting to the first {N_CUTOFF} entries")
df = df.head(N_CUTOFF)

print("Reading vectors")
model = KeyedVectors.load_word2vec_format(args.vectors_path)

sample = {}

def sample_word(df):
    return df.sample(n=1).iloc[0]

print("Sampling words")
while len(sample) < 1000:
    sampled_row = sample_word(df)
    word = sampled_row["TOKEN"]
    frequency = sampled_row["TOT"]
    # If word was already sampled by chance, try again
    if (frequency, word) in sample:
        continue

    # If no vector is available, try again
    if word not in model.key_to_index:
        continue

    vector = model[word].tolist()
    sample[(frequency, word)] = (vector)

print("Sorting by frequency")
keys = sorted(sample.keys(), reverse=True)

output_vectors = ""
for key in keys:
    vector = sample[key]
    frequency, word = key

    vector_as_text = word + " " +  str(frequency) + " " + " ".join(map(str, vector))
    output_vectors += vector_as_text + "\n"

# Turn into valid w2v type
output_text = f"{str(len(sample))} {model.vector_size}\n{output_vectors}"

# Output to file
with open("vectors.txt", "wt") as writer:
    writer.write(output_text)