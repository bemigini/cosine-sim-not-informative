import math
import itertools
import numpy as np
import matplotlib.pyplot as plt


# Compute all maximal minors of W
def maximal_minors(W, eps=1e-10):
    N, D = W.shape
    minors = np.zeros(math.comb(N, D))
    for i, rows in enumerate(itertools.combinations(range(N), D)):
        minors[i] = np.linalg.det(W[list(rows), :])
    # We replace all very small determinant values with zero
    minors[np.abs(minors) < eps] = 0.
    return minors


def braid(n):
    B = np.zeros((math.comb(n, 2), n))
    for i, (l, r) in enumerate(itertools.combinations(range(n), 2)):
        B[i, l] = 1
        B[i, r] = -1
    return B


def colspace(W, eps=1e-10):
    N, D = W.shape
    assert N >= D
    L, s, R = np.linalg.svd(W, full_matrices=True)
    rank = np.sum(s > eps)
    assert rank <= D
    L = L[:, :rank] * np.sqrt(s[:rank])
    return L


def to_point_configuration(A):
    N, D = A.shape
    # We add a constant column
    # This makes the sign pattern of A equivalent to
    # all possible ways an affine hyperplane can classify the rows of A
    # into positive and negative
    return np.hstack([np.ones((N, 1)), A])


def estimate_sign_pattern(W, num_samples=10000):
    N, D = W.shape
    X = np.random.normal(0, 1, (num_samples, D))
    sign_pat = np.unique(np.sign(X @ W.T), axis=0)
    sign_pat = set(tuple(r) for r in sign_pat.tolist())
    return sign_pat


def estimate_ranking_pattern(W, num_samples=10000):
    N, D = W.shape
    X = np.random.normal(0, 1, (num_samples, D))
    rank_pat = np.argsort(X @ W.T) + 1
    rank_pat = set(tuple(r) for r in rank_pat.tolist())
    return rank_pat


def print_matrix_rows(A):
    s = "\\def\\rows{%\n"
    rows = []
    for row in A:
        content = ", ".join(f"{r:.2f}" for r in row)
        rows.append("{%s}" % content)
    s += ",%\n".join(rows)
    s += "}\n"
    return s


def vec_to_sign_vec(v):
    parts = []
    for c in v:
        if c == 1:
            parts.append("+")
        elif c == -1:
            parts.append("-")
        elif c == 0:
            parts.append("0")
        else:
            raise ValueError("Unexpected value in sign vector")
    return "".join(parts)


if __name__ == "__main__":

    # This code proves that the sign pattern does not determine the
    # ranking pattern by providing an example of two point configurations with
    # the same sign pattern but a different ranking pattern.

    EPS = 1e-10

    a_1 = np.array([-.8, 1])
    a_5 = np.array([1, -1])
    A = np.array([a_1,
                  [.8, 1],
                  .25 * a_1 + .75 * a_5,
                  [-1, -1],
                  a_5,
                  ])

    b_1 = np.array([-1, 1])
    b_5 = np.array([.8, -1])
    B = np.array([b_1,
                  [1, 1],
                  .25 * b_1 + .75 * b_5,
                  [-.8, -1],
                  b_5,
                  ])

    # plt.scatter(*A.T)
    # plt.show()
    # plt.scatter(*B.T)
    # plt.show()

    N, D = A.shape

    Ab = to_point_configuration(A)
    Bb = to_point_configuration(B)

    print(Bb @ Bb.T)

    print(f"A =\n{print_matrix_rows(Ab)}")
    print(f"B =\n{print_matrix_rows(Bb)}")

    signpat_A = estimate_sign_pattern(Ab)
    signpat_B = estimate_sign_pattern(Bb)
    assert signpat_A == signpat_B
    print("The sign pattern is:")
    for sv in sorted(signpat_A):
        print("\\tsv{%s}" % vec_to_sign_vec(sv))
    print()

    # Equivalent to check above, but more efficient
    mmab = np.sign(maximal_minors(Ab))
    mmbb = np.sign(maximal_minors(Bb))
    assert (np.all(mmab == mmbb))

    # Compute the rankings possible (hacky, but works for few points in low dimensions)
    # Take a gaussian point cloud and classify all points, restrict to unique
    rankings_A = estimate_ranking_pattern(A)
    rankings_B = estimate_ranking_pattern(B)
    print(rankings_A - rankings_B)
    print(rankings_B - rankings_A)

    # Compute Maximal Minors
    Ca = colspace(braid(N) @ Ab, eps=EPS)
    Cb = colspace(braid(N) @ Bb, eps=EPS)

    mmca = maximal_minors(Ca)
    mmcb = maximal_minors(Cb)

    smmca = np.sign(mmca)
    smmcb = np.sign(mmcb)
    assert not np.all(smmca == smmcb)
    print(f"The Ranking pattern disagrees at a single minor that has magnitude {mmca[smmca != smmcb]} vs {mmcb[smmca != smmcb]}")
