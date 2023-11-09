import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import argparse


def mandelbrot_value(point, max_iters):
    x, y = point
    Z_0 = complex(0, 0)
    C = complex(x, y)
    iters = 0
    while abs(Z_0) < 2 and iters < max_iters:
        Z_0 = (Z_0 * Z_0) + C
        iters += 1
    if iters == max_iters:
        return 0
    return iters


def compute_row(y, x_values, max_iters):
    return [mandelbrot_value((x, y), max_iters) for x in x_values]


def mandelbrot(space=(-2, 2), max_iters=50, eps=1e-3):
    x_values = np.arange(space[0], space[1], eps)
    y_values = np.arange(space[0], space[1], eps)


    with Pool() as pool:
        complex_plane = pool.starmap(compute_row, [(y, x_values, max_iters) for y in y_values])

    return np.array(complex_plane)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eps', type=float, default=1e-3, help='The value that Z has to reach before terminating')
    parser.add_argument('--max_iters', type=int, default=50, help='The max amount of iterations before termination')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mb_set = mandelbrot(max_iters=args.max_iters, eps=args.eps)
    plt.imshow(mb_set)
    plt.show()
