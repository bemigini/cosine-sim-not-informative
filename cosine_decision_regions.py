import math
import argparse
import itertools
import numpy as np


# Compute all maximal minors of W
def maximal_minors(W, eps=1e-10):
    N, D = W.shape
    minors = np.zeros(math.comb(N, D))
    for i, rows in enumerate(itertools.combinations(range(N), D)):
        minors[i] = np.linalg.det(W[list(rows), :])
    # We replace all very small determinant values with zero
    minors[np.abs(minors) < eps] = 0.
    return minors


def colspace(W, eps=1e-10):
    N, D = W.shape
    assert N >= D
    L, s, R = np.linalg.svd(W, full_matrices=True)
    rank = np.sum(s > eps)
    assert rank <= D
    L = L[:, :rank] * np.sqrt(s[:rank])
    return L


def braid(n):
    B = np.zeros((math.comb(n, 2), n))
    for i, (l, r) in enumerate(itertools.combinations(range(n), 2)):
        B[i, l] = 1
        B[i, r] = -1
    return B


def gram_matrix(W):
    return W @ W.T


def cosine_gram_matrix(W):
    D = np.linalg.inv(np.diag(np.linalg.norm(W, axis=1)))
    Wc = D @ W
    return Wc @ Wc.T


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--d', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-cosine', action='store_true')
    parser.add_argument('--use-braid', action='store_true')

    args = parser.parse_args()
    np.random.seed(args.seed)

    EPS = 1e-10

    W = np.random.normal(0, 1, (args.n, args.d))

    if args.use_braid:
        B = braid(args.n)
        W = B @ W

    if args.use_cosine:
        G = cosine_gram_matrix(W)
    else:
        G = gram_matrix(W)

    L = colspace(G, eps=EPS)

    maxm_W = np.sign(np.array(maximal_minors(W)))
    maxm_L = np.sign(np.array(maximal_minors(L)))

    if maxm_L[0] != maxm_W[0]:
        maxm_L = -maxm_L

    assert np.allclose(maxm_L, maxm_W)
