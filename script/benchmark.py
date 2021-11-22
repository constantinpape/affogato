import argparse
import time

import h5py
import pandas as pd
import numpy as np
from affogato.segmentation import compute_mws_segmentation

# The example data from https://oc.embl.de/index.php/s/sXJzYVK0xEgowOz
PATH = "/home/pape/Work/data/isbi/isbi_test_volume.h5"

OFFSETS = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
           [-1, -1, -1], [-1, 1, 1], [-1, -1, 1], [-1, 1, -1],
           [0, -9, 0], [0, 0, -9],
           [0, -9, -9], [0, 9, -9], [0, -9, -4], [0, -4, -9], [0, 4, -9], [0, 9, -4],
           [0, -27, 0], [0, 0, -27]]


def measure_runtime(affs, n):
    times = []
    for _ in range(n):
        t = time.time()
        compute_mws_segmentation(affs, OFFSETS, 3)
        times.append(time.time() - t)
    return np.min(times)


def increase_shape(shape, full_shape, axis, is_full):
    if all(is_full):
        return shape, axis, is_full
    if is_full[axis]:
        axis = (axis + 1) % 3
        return increase_shape(shape, full_shape, axis, is_full)

    shape[axis] *= 2
    if shape[axis] >= full_shape[axis]:
        is_full[axis] = True
        shape[axis] = full_shape[axis]
    axis = (axis + 1) % 3
    return shape, axis, is_full


def benchmark(path, out_path, n):
    with h5py.File(path, "r") as f:
        affs = f["affinities"][:]
    assert affs.shape[0] == len(OFFSETS)
    seperating_channel = 3
    affs[:seperating_channel] *= -1
    affs[:seperating_channel] += 1

    full_shape = affs.shape[1:]
    shape = [4, 64, 64]

    results = []
    axis = 0
    is_full = [False] * 3

    while True:
        bb = (slice(None),) + tuple(slice(0, sh) for sh in shape)
        affs_ = affs[bb]
        t = measure_runtime(affs_, n)
        print("Run benchmark for", shape, "in t", t, "[s]")
        str_shape = "-".join(map(str, shape))
        size = np.prod(list(shape))
        results.append([str_shape, size, t])

        shape, axis, is_full = increase_shape(shape, full_shape, axis, is_full)
        if all(is_full):
            break

    t = measure_runtime(affs, n)
    print("Run benchmark for", full_shape, "in t", t, "[s]")
    str_shape = "-".join(map(str, full_shape))
    size = np.prod(list(full_shape))
    results.append([str_shape, size, t])

    results = pd.DataFrame(results, columns=["shape", "size", "time [s]"])
    results.to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_path", required=True)  # where to save the csv with results
    parser.add_argument("-p", "--path", default=PATH)
    parser.add_argument("-n", default=5, type=int)
    args = parser.parse_args()
    benchmark(args.path, args.output_path, args.n)


if __name__ == "__main__":
    main()
