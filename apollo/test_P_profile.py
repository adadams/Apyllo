import numpy as np


def P1(minP, maxP, natm):
    return maxP + (minP - maxP) * np.arange(natm) / (natm - 1)


def P2(minP, maxP, natm):
    return np.linspace(maxP, minP, natm)


print(np.isclose(P1(-4, 2.5, 71), P2(-4, 2.5, 71), rtol=1e-5))
