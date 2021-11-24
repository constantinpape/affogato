import argparse
import os

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def parse_result(path):
    tab = pd.read_csv(path)[["size", "time [s]"]]
    return tab


def plot_benchmark(inputs):
    results = []
    names = []
    for inp in inputs:
        names.append(os.path.splitext(os.path.split(inp)[1])[0])
        results.append(parse_result(inp))

    for name, res in zip(names, results):
        sns.lineplot(data=res, x="size", y="time [s]", label=name)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", "-i", required=True, nargs="+")
    args = parser.parse_args()
    plot_benchmark(args.inputs)


if __name__ == "__main__":
    main()
