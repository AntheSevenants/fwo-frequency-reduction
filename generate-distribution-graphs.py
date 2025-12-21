import argparse
import math
import model.helpers
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    description='generate-distribution-graphs - zulke schone grafiekskes, en in kleur')
parser.add_argument("type", help="evaluation | exponential")
args = parser.parse_args()

MAX_FREQ = 100
SAMPLE_SIZE = 100

x = list(range(1, SAMPLE_SIZE + 1, 1))

if args.type == "evaluation":
    zipfian_frequencies = model.helpers.generate_zipfian_frequencies(
        MAX_FREQ,
        SAMPLE_SIZE,
        zipf_param=1.0)
    flat_frequencies = [ 1 ] * SAMPLE_SIZE

    zipfian_probabilities = np.divide(zipfian_frequencies, np.sum(zipfian_frequencies))
    flat_probabilities = np.divide(flat_frequencies, np.sum(flat_frequencies))

    # Boolean mask where Zipfian becomes smaller than flat
    mask = zipfian_probabilities < flat_probabilities
    cross_idx = np.nonzero(mask)[0][0]

    # Split into first and second halves
    zipf_first  = zipfian_probabilities[:cross_idx]
    zipf_second = zipfian_probabilities[cross_idx:]

    width = 1

    plt.bar(x[:cross_idx], zipf_first, label="Zipfian distribution", color="C0")
    plt.bar(x, flat_probabilities, label="Uniform distribution", color="C1")
    plt.bar(x[cross_idx:], zipf_second, label="_Zipfian distribution", color="C0")
    plt.tight_layout()

    plt.legend()

    filename = "fig-evaluation-probabilities"
elif args.type == "exponential":
    exponential_frequencies_1 = model.helpers.generate_exponential_frequencies(
        MAX_FREQ,
        SAMPLE_SIZE,
        exp_param=math.log(1, 10)
    )
    exponential_frequencies_1 = np.divide(
        exponential_frequencies_1,
        np.sum(exponential_frequencies_1))
    exponential_frequencies_2 = model.helpers.generate_exponential_frequencies(
        MAX_FREQ,
        SAMPLE_SIZE,
        exp_param=math.log(2, 10)
    )
    exponential_frequencies_2 = np.divide(
        exponential_frequencies_2,
        np.sum(exponential_frequencies_2))
    exponential_frequencies_3 = model.helpers.generate_exponential_frequencies(
        MAX_FREQ,
        SAMPLE_SIZE,
        exp_param=math.log(3, 10)
    )
    exponential_frequencies_3 = np.divide(
        exponential_frequencies_3,
        np.sum(exponential_frequencies_3))

    # First intersection
    mask = exponential_frequencies_3 < exponential_frequencies_2
    cross_idx = np.nonzero(mask)[0][0]

    # Second intersection
    mask_2 = exponential_frequencies_3 < exponential_frequencies_1
    cross_idx_2 = np.nonzero(mask_2)[0][0]

    # Third intersection
    mask_3 = exponential_frequencies_2 < exponential_frequencies_1
    cross_idx_3 = np.nonzero(mask_3)[0][0]

    # Stuk 1: groen eerst, geel tweede, blauw laatste
    plt.bar(x[:cross_idx], exponential_frequencies_3[:cross_idx], color="C2")
    plt.bar(x[:cross_idx], exponential_frequencies_2[:cross_idx], color="C1")  
    
    # Stuk 2: geel eerst, groen tweede, blauw laatste
    plt.bar(x[cross_idx:cross_idx_2],
            exponential_frequencies_2[cross_idx:cross_idx_2], color="C1")
    plt.bar(x[cross_idx:cross_idx_2],
            exponential_frequencies_3[cross_idx:cross_idx_2], color="C2")    
    
    # Stuk 3: geel eerst, blauw tweede, groen laatste
    plt.bar(x[cross_idx_2:cross_idx_3],
            exponential_frequencies_2[cross_idx_2:cross_idx_3],
            color="C1")
    plt.bar(x[:cross_idx_3], exponential_frequencies_1[:cross_idx_3], color="C0")
    plt.bar(x[cross_idx_2:cross_idx_3],
            exponential_frequencies_3[cross_idx_2:cross_idx_3],
            color="C2")

    # Stuk 4: blauw eerst, geel tweede, groen laatste
    plt.bar(x[cross_idx_3:], exponential_frequencies_1[cross_idx_3:], color="C0",
            label="log(1)")
    plt.bar(x[cross_idx_3:], exponential_frequencies_2[cross_idx_3:], color="C1",
            label="log(2)")
    plt.bar(x[cross_idx_3:], exponential_frequencies_3[cross_idx_3:], color="C2",
            label="log(3)")
   
    filename = "fig-exponential-probabilities-2"
elif args.type == "memory-zipfian":
    zipfian_memory = model.helpers.zipf_exemplars_per_construction(
        1000,
        100,
        1.1,
        5
    )
    plt.bar(range(1, len(zipfian_memory) + 1), zipfian_memory)
    
    filename = "fig-memory-zipfian"
elif args.type == "memory-exponential-zero":
    exponential_memory = model.helpers.exp_exemplars_per_construction(
        1000,
        100,
        0,
        5
    )
    plt.bar(range(1, len(exponential_memory) + 1), exponential_memory)

    filename = "fig-memory-exponential"
elif args.type == "distribution-comparison":
    N_SAMPLE = 100
    N_LARGE = 130000
    zipfian_sample = model.helpers.generate_zipfian_sample(
        n_sample=N_SAMPLE,
        n_large=N_LARGE
    )
    exponential_sample = model.helpers.generate_exponential_sample(
        n_sample=N_SAMPLE,
        n_large=N_LARGE,
        exp_param=math.log(2.8, 10)
    )
    zipfian_sample = [ 1 - item[1] for item in zipfian_sample ]
    exponential_sample = [ 1 - item[1] for item in exponential_sample ]

    plt.plot(zipfian_sample, c="C0", label="Zipfian distribution")
    plt.plot(exponential_sample, c="C1", label="Exponential distribution")

    filename = "fig-distribution-comparison"
else:
    raise ValueError("Unknown graph type")

plt.tight_layout()
plt.legend()

plt.savefig(f"figures/{filename}.png")