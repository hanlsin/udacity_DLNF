import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.


def cross_entropy(Y, P):
    # yiln(pi)
    ylnp = Y * np.log(P)
    # (1-yi)ln(1-pi)
    oylnp = np.subtract(1, Y) * np.log(np.subtract(1, P))
    return np.sum(ylnp + oylnp) * -1


Y = np.array([1, 0, 1, 1])
P = np.array([0.4, 0.6, 0.1, 0.5])
print(cross_entropy(Y, P))
