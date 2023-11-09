import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
from tqdm import tqdm


def mandelbrot(z_0=None, space=(-2, 2), max_iters=50, eps=1e-3, verbose=True):
    x_values = np.arange(space[0], space[1], eps)
    y_values = np.arange(space[0], space[1], eps)
    complex_plane = []

    if verbose:
        x_values = tqdm(x_values)

    for x in x_values:
        complex_plane.append([])
        for y in y_values:
            if z_0 is None:
                Z_0 = complex(0, 0)
            else:
                Z_0 = complex(*z_0)
            C = complex(x, y)
            iters = 0
            while abs(Z_0) < 2 and iters < max_iters:
                Z_0 = (Z_0 * Z_0) + C
                iters += 1
            if iter == max_iters:
                complex_plane[-1].append(0)
            else:
                complex_plane[-1].append(iters)
    return np.array(complex_plane)


def find_circle_points(space=(-2, 2), eps=1e-3):
    x_values = np.arange(space[0], space[1], eps)
    y_values = np.arange(space[0], space[1], eps)
    tolerance = (np.sqrt(2) * eps) / 2
    lower_bound = 1 - tolerance
    upper_bound = 1 + tolerance
    circle = [(x, y) for x in x_values for y in y_values if lower_bound**2 <= (x ** 2) + (y ** 2) <= upper_bound**2]
    circle_points = [(x, y, np.arctan2(y, x)) for x, y in circle]
    circle_points = sorted(circle_points, key=lambda point: point[2])
    circle_points = [(x, y) for x, y, angle in circle_points]
    return circle_points


def julia(z_0=None, space=(-2, 2), max_iters=50, eps=1e-3):

    circle = find_circle_points(space=space, eps=eps)

    frames = []
    for coords in tqdm(circle):
        frames.append(
            mandelbrot(
                z_0=coords,
                space=space,
                max_iters=max_iters,
                eps=eps,
                verbose=False,
                )
            )

    fig, ax = plt.subplots()
    im = ax.imshow(frames[0], animated=True)

    def update(frame):
        im.set_array(frame)
        return [im]

    anim = FuncAnimation(fig, update, frames=frames, blit=True)
    anim.save('julia_animation.mp4', writer='ffmpeg', fps=30)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eps', type=float, default=1e-3, help='The value that Z has to reach before terminating')
    parser.add_argument('--max_iters', type=int, default=50, help='The max amount of iterations before termination')
    parser.add_argument('--julia', action='store_true', help='Draw a julia animation')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.julia:
        julia(max_iters=args.max_iters, eps=args.eps)
    else:
        mb_set = mandelbrot(max_iters=args.max_iters, eps=args.eps)
        plt.imshow(mb_set)
        plt.show()
