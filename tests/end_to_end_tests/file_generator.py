#!/bin/env python3
import numpy as np
import sys

if len(sys.argv) != 5:
    print(
        f"usage: {
            sys.argv[0]} number_of_samples",
        "max_repeats max_val output_filename",
    )
    exit(1)
size = int(sys.argv[1])
repeats = np.random.randint(1, int(sys.argv[2]), size)
choices = np.random.choice(list(range(int(sys.argv[3]))), size)
output = np.repeat(choices, repeats).astype("B")
output.tofile(sys.argv[4])
