# --------------------------------------------------------
# Written by Yufei Ye and modified by Sheng-Yu Wang (https://github.com/JudyYe)
# Convert from MATLAB code https://inst.eecs.berkeley.edu/~cs194-26/fa18/hw/proj3/gradient_starter.zip
# --------------------------------------------------------
from __future__ import print_function

import argparse
import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg

def toy_recon(im):
    imh, imw = im.shape
    im2var = np.arange(imh * imw).reshape((imh, imw)).astype(int)

    constraints = []
    targets = []
    source = im
    for y in range(imh):
        constraint_matrix = np.zeros((imw - 1, imh * imw))
        constraint_matrix[np.arange(imw-1), np.roll(im2var[y], -1)[:-1]] = 1
        constraint_matrix[np.arange(imw-1), im2var[y][:-1]] = -1
        target_matrix = (np.roll(source[y], -1) - source[y])[:-1]
        constraints.append(constraint_matrix)
        targets.append(target_matrix)

    for x in range(imw):
        constraint_matrix = np.zeros((imh - 1, imh * imw))
        constraint_matrix[np.arange(imh-1), np.roll(im2var[:, x], -1)[:-1]] = 1
        constraint_matrix[np.arange(imh-1), im2var[:, x][:-1]] = -1
        target_matrix = (np.roll(source[:, x], -1) - source[:, x])[:-1] # or np.diff
        constraints.append(constraint_matrix)
        targets.append(target_matrix)

    # Pixel constraint
    constraint_matrix = np.zeros((1, imh * imw))
    constraint_matrix[0, im2var[0][0]] = 1
    target_matrix = np.array([source[0][0]])

    constraints.append(constraint_matrix)
    targets.append(target_matrix)

    constraint_matrix = np.concatenate(constraints, axis=0)
    target_matrix = np.concatenate(targets, axis=0)

    # Build a sparse matrix
    constraint_matrix = sparse.csr_matrix(constraint_matrix)

    # Solve
    result = linalg.lsqr(constraint_matrix, target_matrix)[0]
    result = result.reshape((imh, imw))

    return result


def poisson_blend(fg, mask, bg):
    """
    Poisson Blending.
    :param fg: (H, W, C) source texture / foreground object
    :param mask: (H, W, 1)
    :param bg: (H, W, C) target image / background
    :return: (H, W, C)
    """
    outs = []
    for c in range(fg.shape[2]):
        outs.append(_blend_channel(fg[:, :, c], mask[..., 0], bg[:, :, c]))

    out = np.stack(outs, axis=2)
    # rescale
    # out = (out - min(out.min(), 0)) / (max(out.max(), 1) - min(out.min(), 0))

    return out

# def _blend_channel(fg, mask, bg, rescale=True):
def _blend_channel(fg, mask, bg, rescale=False):
    """
    Blend a single channel.
    :param fg: (H, W) source texture / foreground object
    :param mask: (H, W), 1 for foreground, 0 for background
    :param bg: (H, W) target image / background
    :return: (H, W)
    """
    constraints = []
    targets = []

    fg_h, fg_w = fg.shape
    fg2var = np.arange(fg_h * fg_w).reshape((fg_h, fg_w)).astype(int)

    def extract_boundary_updates(constraint_matrix, out_size):
        # assumes constraint matrix (Eq x Pixels) has been subsetted such that there is at most one active variable in each row
        # returns a vector of updates to the target matrix
        constraint_x, constraint_y, constraint_values = sparse.find(constraint_matrix)
        target_updates = constraint_values * bg.flatten()[constraint_y]
        target_mask = ~mask.flatten()[constraint_y]
        target_updates_values = target_updates[target_mask]
        target_update_x = constraint_x[target_mask]
        target_updates = np.zeros(out_size)
        target_updates[target_update_x] = -target_updates_values
        return target_updates

    # Construct COO Sparse matrix from gradient constraints
    var_y = mask.sum(1)
    for y in range(fg_h):
        if var_y[y] == 0:
            continue
        coord_x = np.arange(fg_w-1) # constraint count
        coord_y_left = fg2var[y][:-1] # v_j
        coord_y_right = np.roll(fg2var[y], -1)[:-1] # v_i
        coo_left = sparse.coo_matrix((np.full(coord_x.shape, -1), (coord_x, coord_y_left)), shape=(fg_w-1, fg_w * fg_h))
        coo_right = sparse.coo_matrix((np.full(coord_x.shape, 1), (coord_x, coord_y_right)), shape=(fg_w-1, fg_w * fg_h))
        constraint_matrix = coo_left + coo_right
        constraint_matrix = constraint_matrix[mask[y, :-1], :] # only keep constraints with variables

        target_updates = extract_boundary_updates(constraint_matrix, mask[y,:-1].sum())
        constraint_matrix = constraint_matrix[:, mask.flatten()] # only optimize with other variables
        constraints.append(constraint_matrix)

        target_matrix = (np.roll(fg[y], -1) - fg[y])[:-1][mask[y, :-1]]
        target_matrix = target_matrix + target_updates
        targets.append(target_matrix)

        # Now do the other direction
        # Actually, the only constraints that are different are the ones that are on the boundary
        # For masking logic to work, our _reference_ must be the right coordinate this time
        coord_y_right = fg2var[y][1:] # v_i
        coord_y_left = np.roll(fg2var[y], 1)[1:] # v_j
        coo_left = sparse.coo_matrix((np.full(coord_x.shape, 1), (coord_x, coord_y_left)), shape=(fg_w-1, fg_w * fg_h))
        coo_right = sparse.coo_matrix((np.full(coord_x.shape, -1), (coord_x, coord_y_right)), shape=(fg_w-1, fg_w * fg_h))
        constraint_matrix = coo_left + coo_right
        constraint_matrix = constraint_matrix[mask[y, 1:], :] # only keep constraints with variables

        target_updates = extract_boundary_updates(constraint_matrix, mask[y,1:].sum())
        constraint_matrix = constraint_matrix[:, mask.flatten()] # only optimize with other variables
        constraint_matrix = constraint_matrix[target_updates != 0]
        constraints.append(constraint_matrix)

        target_matrix = (fg[y] - np.roll(fg[y], 1))[1:][mask[y, 1:]]
        target_matrix = target_matrix + target_updates
        target_matrix = target_matrix[target_updates != 0]

        targets.append(target_matrix)

    var_x = mask.sum(0)
    for x in range(fg_w):
        if var_x[x] == 0:
            continue
        coord_x = np.arange(fg_h-1)
        coord_y_left = fg2var[:, x][:-1]
        coord_y_right = np.roll(fg2var[:, x], -1)[:-1]
        coo_left = sparse.coo_matrix((np.full(coord_x.shape, -1), (coord_x, coord_y_left)), shape=(fg_h-1, fg_w * fg_h))
        coo_right = sparse.coo_matrix((np.full(coord_x.shape, 1), (coord_x, coord_y_right)), shape=(fg_h-1, fg_w * fg_h))
        constraint_matrix = coo_left + coo_right
        constraint_matrix = constraint_matrix[mask[:-1, x], :] # only keep constraints with variables

        target_updates = extract_boundary_updates(constraint_matrix, mask[:-1,x].sum())

        constraint_matrix = constraint_matrix[:, mask.flatten()] # only optimize with other variables
        constraints.append(constraint_matrix)

        target_matrix = (np.roll(fg[:, x], -1) - fg[:, x])[:-1][mask[:-1, x]]
        target_matrix = target_matrix + target_updates
        targets.append(target_matrix)

        # Now do the other direction
        coord_y_right = fg2var[:, x][1:]
        coord_y_left = np.roll(fg2var[:, x], 1)[1:]
        coo_left = sparse.coo_matrix((np.full(coord_x.shape, 1), (coord_x, coord_y_left)), shape=(fg_h-1, fg_w * fg_h))
        coo_right = sparse.coo_matrix((np.full(coord_x.shape, -1), (coord_x, coord_y_right)), shape=(fg_h-1, fg_w * fg_h))
        constraint_matrix = coo_left + coo_right
        constraint_matrix = constraint_matrix[mask[1:, x], :] # only keep constraints with variables

        target_updates = extract_boundary_updates(constraint_matrix, mask[1:,x].sum())
        constraint_matrix = constraint_matrix[:, mask.flatten()] # only optimize with other variables
        # Only keep boundary updates
        constraint_matrix = constraint_matrix[target_updates != 0]
        constraints.append(constraint_matrix)

        target_matrix = (fg[:, x] - np.roll(fg[:, x],1))[1:][mask[1:, x]]
        target_matrix = target_matrix + target_updates
        target_matrix = target_matrix[target_updates != 0]
        targets.append(target_matrix)

    constraint_matrix = sparse.vstack(constraints)
    target_matrix = np.concatenate(targets, axis=0)

    # Solve
    result = linalg.lsqr(constraint_matrix, target_matrix)[0]
    fg_new = fg.copy()
    fg_new[mask] = result

    if rescale:
        # if max < 1, use 1, if max > 1, use max
        new_max = max(result.max(), 1)
        new_min = min(result.min(), 0)
        fg_new = (fg_new - new_min) / (new_max - new_min)
    return fg_new * mask + bg * (1 - mask)

def mixed_blend(fg, mask, bg):
    """EC: Mix gradient of source and target"""
    return fg * mask + bg * (1 - mask)


def color2gray(rgb_image):
    """Naive conversion from an RGB image to a gray image."""
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)


def mixed_grad_color2gray(rgb_image):
    """EC: Convert an RGB image to gray image using mixed gradients."""
    return np.zeros_like(rgb_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Poisson blending.")
    parser.add_argument("-q", "--question", default="blend", choices=["toy", "blend", "mixed", "color2gray"])
    # parser.add_argument("-q", "--question", required=True, choices=["toy", "blend", "mixed", "color2gray"])
    args, _ = parser.parse_known_args()

    # Example script: python proj2_starter.py -q toy
    if args.question == "toy":
        image = imageio.imread('./data/toy_problem.png') / 255.
        image_hat = toy_recon(image)

        plt.subplot(121)
        plt.imshow(image, cmap='gray')
        plt.title('Input')
        plt.subplot(122)
        plt.imshow(image_hat, cmap='gray')
        plt.title('Output')
        plt.show()

    # Example script: python proj2_starter.py -q blend -s data/source_01_newsource.png -t data/target_01.jpg -m data/target_01_mask.png
    if args.question == "blend":
        parser.add_argument("-s", "--source", required=True)
        parser.add_argument("-t", "--target", required=True)
        parser.add_argument("-m", "--mask", required=True)
        args = parser.parse_args()

        # after alignment (masking_code.py)
        ratio = 1
        fg = cv2.resize(imageio.imread(args.source), (0, 0), fx=ratio, fy=ratio)
        bg = cv2.resize(imageio.imread(args.target), (0, 0), fx=ratio, fy=ratio)
        mask = cv2.resize(imageio.imread(args.mask), (0, 0), fx=ratio, fy=ratio)

        fg = fg / 255.
        bg = bg / 255.
        mask = (mask.sum(axis=2, keepdims=True) > 0)

        blend_img = poisson_blend(fg, mask, bg)

        plt.subplot(121)
        plt.imshow(fg * mask + bg * (1 - mask))
        plt.title('Naive Blend')
        plt.subplot(122)
        plt.imshow(blend_img)
        plt.title('Poisson Blend')
        plt.show()

    if args.question == "mixed":
        parser.add_argument("-s", "--source", required=True)
        parser.add_argument("-t", "--target", required=True)
        parser.add_argument("-m", "--mask", required=True)
        args = parser.parse_args()

        # after alignment (masking_code.py)
        ratio = 1
        fg = cv2.resize(imageio.imread(args.source), (0, 0), fx=ratio, fy=ratio)
        bg = cv2.resize(imageio.imread(args.target), (0, 0), fx=ratio, fy=ratio)
        mask = cv2.resize(imageio.imread(args.mask), (0, 0), fx=ratio, fy=ratio)

        fg = fg / 255.
        bg = bg / 255.
        mask = (mask.sum(axis=2, keepdims=True) > 0)

        blend_img = mixed_blend(fg, mask, bg)

        plt.subplot(121)
        plt.imshow(fg * mask + bg * (1 - mask))
        plt.title('Naive Blend')
        plt.subplot(122)
        plt.imshow(blend_img)
        plt.title('Mixed Blend')
        plt.show()

    if args.question == "color2gray":
        parser.add_argument("-s", "--source", required=True)
        args = parser.parse_args()

        rgb_image = imageio.imread(args.source)
        gray_image = color2gray(rgb_image)
        mixed_grad_img = mixed_grad_color2gray(rgb_image)

        plt.subplot(121)
        plt.imshow(gray_image, cmap='gray')
        plt.title('rgb2gray')
        plt.subplot(122)
        plt.imshow(mixed_grad_img, cmap='gray')
        plt.title('mixed gradient')
        plt.show()

    plt.close()
