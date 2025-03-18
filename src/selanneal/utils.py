import matplotlib.pyplot as plt
import numpy as np


def ranking(features, population, rank_threshold=500):
    """
    rank_threshold: cutoff rank of population when sorted according to energy
    """
    # sort population dict after energy
    population_list = sorted(population.items(), key=lambda item: item[1][0])[:rank_threshold]
    feat_indices = [list(key) for key, val in population_list]
    frequencies, _ = np.histogram(np.array(feat_indices).flatten(), bins=len(features), range=(0, len(features)))

    argsorted = np.argsort(frequencies)[::-1]

    plt.plot(frequencies[argsorted], marker='x', ls='', ms=7)
    sorted_features = list(np.asarray(features)[argsorted])
    plt.xticks(range(len(features)), sorted_features, rotation=90)
    plt.ylabel('frequency')
    plt.ylim(0, None)
    plt.grid(axis='x')

    plt.gca().set_axisbelow(True)
    plt.minorticks_off()

    return sorted_features
