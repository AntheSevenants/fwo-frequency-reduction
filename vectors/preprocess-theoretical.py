import argparse
import os
import random
import sys
import math
import pandas as pd

from gensim.models import KeyedVectors

from helpers import generate_zipfian_sample

parser = argparse.ArgumentParser(description='preprocess - sample words from a CGN frequency list and associate them with vectors')
parser.add_argument('frequency_path', type=str, help='the file path to your COREX frequency list')
parser.add_argument('sample_n', type=int, help='the number of words to sample')
parser.add_argument('output_path', type=str, help='the output path for the dataset')
args = parser.parse_args()

N_CUTOFF = 10000

if args.sample_n > N_CUTOFF:
    raise ValueError(f"Sample size is bigger than cutoff size ({N_CUTOFF})")

print("Reading frequency list")
df = pd.read_csv(args.frequency_path, sep="\t")
print("Removing tagged entries")
df = df[df['tag'].isna()]
print("Removing non-alphanumeric entries")
df = df[df.TOKEN.str.isalpha()]
print("Filtering other weird tokens")
df = df[~df.TOKEN.isin(["xxx", "ggg"])]
print("Computing cumulative sums")
df['cumulative_sum'] = df['TOT'].cumsum()
df['RANK'] = df["RANK"].rank()

max_frequency = df["TOT"].max()
total_frequency_real = df["TOT"].sum()

df['percentile'] = df.apply(lambda row: row.cumulative_sum / total_frequency_real, axis=1)

# print(f"Limiting to the first {N_CUTOFF} entries")
# df = df.head(N_CUTOFF)

print("Generating theoretical percentiles")
theoretical_percentiles = generate_zipfian_sample(n_sample=args.sample_n, zipf_param=0.9)

columns = ["rank", "theoretical_percentile"]
sampled_words = pd.DataFrame.from_records(theoretical_percentiles, columns=columns)

print("Cross-referencing with typerank file")
sampled_words = sampled_words.merge(df, left_on="rank", right_on="RANK", how="left")

print("Sorting by frequency")
sampled_words = sampled_words.sort_values(by=["theoretical_percentile"], ascending=True)

print("Renaming columns")
sampled_words = sampled_words.rename(columns={'RANK': 'rank', 'TOKEN': 'token'})

print("Computing theoretical frequencies")
sampled_words['frequency'] = sampled_words.apply(lambda row: math.ceil(row["rank"].values[0] ** (-1.1) * max_frequency), axis=1)

sampled_words.to_csv(f"theoretical-percentile-info-{args.sample_n}.tsv", sep="\t", index=False)