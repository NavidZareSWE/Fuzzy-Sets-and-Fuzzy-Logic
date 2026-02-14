"""
Subtractive Clustering for automatic rule generation.

This module implements the subtractive clustering algorithm following
Chiu (1994) as discussed in Mendel (2017).

Algorithm Summary:
1. Each data point is assigned a potential based on its density of
   surrounding points, using a Gaussian neighbourhood of radius ra.
2. The point with the highest potential is selected as the first
   cluster centre.
3. The potential of all remaining points is reduced by the influence
   of the selected centre (radius rb = squash_factor * ra).
4. Steps 2-3 are repeated until the acceptance/rejection criteria are met.
"""

import numpy as np
from config import CLUSTER_RADIUS, SQUASH_FACTOR, ACCEPT_RATIO, REJECT_RATIO


def subtractive_clustering(data, ra=CLUSTER_RADIUS, squash_factor=SQUASH_FACTOR,
                           accept_ratio=ACCEPT_RATIO, reject_ratio=REJECT_RATIO):
    """
    Perform subtractive clustering on normalised data.

    Parameters:
        data : (N, D) array - each row is a normalised data vector
        ra   : neighbourhood radius
        squash_factor, accept_ratio, reject_ratio : algorithm parameters

    Returns:
        centres : (K, D) array - identified cluster centres
    """
    N, D = data.shape
    rb = squash_factor * ra
    alpha = 4.0 / (ra ** 2)
    beta = 4.0 / (rb ** 2)

    # Step 1: compute initial potentials
    potentials = np.zeros(N)
    for i in range(N):
        dists_sq = np.sum((data - data[i]) ** 2, axis=1)
        potentials[i] = np.sum(np.exp(-alpha * dists_sq))

    centres = []
    first_potential = potentials.max()

    while True:
        best_idx = np.argmax(potentials)
        best_potential = potentials[best_idx]

        if len(centres) == 0:
            # Always accept the first centre
            centres.append(data[best_idx].copy())
            first_potential = best_potential
        else:
            ratio = best_potential / first_potential

            if ratio > accept_ratio:
                centres.append(data[best_idx].copy())
            elif ratio < reject_ratio:
                break
            else:
                # Check shortest distance to existing centres
                d_min = min(
                    np.linalg.norm(data[best_idx] - c) for c in centres
                )
                if (d_min / ra) + (best_potential / first_potential) >= 1.0:
                    centres.append(data[best_idx].copy())
                else:
                    potentials[best_idx] = 0.0
                    continue

        # Reduce potentials around the newly accepted centre
        new_centre = centres[-1]
        dists_sq = np.sum((data - new_centre) ** 2, axis=1)
        potentials -= best_potential * np.exp(-beta * dists_sq)
        potentials = np.maximum(potentials, 0.0)

    return np.array(centres)
