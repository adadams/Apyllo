import numpy as np


def BinSpec(flux, err, wavelo, wavehi, binw):
    blen = (int)(len(flux) / binw)
    binflux = np.zeros(blen)
    binerr = np.zeros(blen)
    ibinlo = np.zeros(blen)
    ibinhi = np.zeros(blen)
    fbinlo = np.zeros(blen)
    fbinhi = np.zeros(blen)
    binlo = np.zeros(blen)
    binhi = np.zeros(blen)

    for i in range(0, blen):
        ibinlo[i] = i * binw
        ibinhi[i] = (i + 1) * binw
        fbinlo[i] = np.modf(ibinlo[i])[0]
        fbinhi[i] = np.modf(ibinhi[i])[0]
        if fbinlo[i] == 0.0:
            ibinlo[i] = ibinlo[i] + 0.000001
            fbinlo[i] = 0.000001
        if fbinhi[i] == 0.0:
            ibinhi[i] = ibinhi[i] + 0.000001
            fbinhi[i] = 0.000001

    for i in range(0, len(ibinhi)):
        binflux[i] = np.sum(
            flux[(int)(np.ceil(ibinlo[i])) : (int)(np.floor(ibinhi[i]))]
        )
        binerr[i] = np.sum(err[(int)(np.ceil(ibinlo[i])) : (int)(np.floor(ibinhi[i]))])

        binflux[i] = binflux[i] + (1.0 - fbinlo[i]) * flux[(int)(np.floor(ibinlo[i]))]
        binerr[i] = binerr[i] + (1.0 - fbinlo[i]) * err[(int)(np.floor(ibinlo[i]))]

        if (int)(np.floor(ibinlo[i])) == (int)(len(flux) - 1):
            binlo[i] = (1.0 - fbinlo[i]) * wavelo[(int)(np.floor(ibinlo[i]))]
        else:
            binlo[i] = (1.0 - fbinlo[i]) * wavelo[(int)(np.floor(ibinlo[i]))] + fbinlo[
                i
            ] * wavelo[(int)(np.floor(ibinlo[i])) + 1]

        if (int)(np.floor(ibinhi[i])) == len(flux):
            binhi[i] = (1.0 - fbinhi[i]) * wavehi[(int)(np.floor(ibinhi[i])) - 1]
        else:
            binhi[i] = (1.0 - fbinhi[i]) * wavehi[
                (int)(np.floor(ibinhi[i])) - 1
            ] + fbinhi[i] * wavehi[(int)(np.floor(ibinhi[i]))]

        if (int)(np.floor(ibinhi[i])) >= len(flux):
            binflux[i] = binflux[i]
            binerr[i] = binerr[i]
        else:
            binflux[i] = binflux[i] + fbinhi[i] * flux[(int)(np.floor(ibinhi[i]))]
            binerr[i] = binerr[i] + fbinhi[i] * err[(int)(np.floor(ibinhi[i]))]

        binflux[i] = binflux[i] / binw
        binerr[i] = binerr[i] / binw
        binerr[i] = binerr[i] / np.sqrt(binw - 1.0)

    return binflux, binerr, binlo, binhi


def ConvSpec(flux, bin_width):
    # The factor of 6 is a somewhat arbitrary choice to get
    # a wide enough window for the Gaussian kernel
    kernel_width = bin_width * 6.0
    stdev = bin_width / 2.35

    kernel_remainder, kernel_integer = np.modf(kernel_width)
    if kernel_integer == 0:
        return flux

    kernel_range = np.arange(kernel_integer) + kernel_remainder - (kernel_width / 2)
    kernel = np.exp(-0.5 * (kernel_range / stdev) ** 2)
    kernel = kernel / np.sum(kernel)

    convflux = np.convolve(flux, kernel, mode="same")

    return convflux
