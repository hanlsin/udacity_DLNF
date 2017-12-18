import numpy as np

# Trying for L=[5,6,7].
# The correct answer is
# [0.090030573170380462, 0.24472847105479764, 0.6652409557748219]


def softmax(L):
    exparr = np.exp(L)
    expsum = np.sum(exparr)
    return exparr / expsum


L = np.array([5, 6, 7])
print(softmax(L))
