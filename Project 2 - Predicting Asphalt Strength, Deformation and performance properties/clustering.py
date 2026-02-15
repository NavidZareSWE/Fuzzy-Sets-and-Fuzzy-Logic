import numpy as np
from config import CLUSTER_RADIUS, SQUASH_FACTOR, ACCEPT_RATIO, REJECT_RATIO

# Read More: https://nl.mathworks.com/help/fuzzy/fuzzy-clustering.html#FP2434


def subtractive_clustering(input_data, cluster_radius=CLUSTER_RADIUS, squash_factor=SQUASH_FACTOR,
                           accept_threshold=ACCEPT_RATIO, reject_threshold=REJECT_RATIO):
    num_samples, num_features = input_data.shape
    neighbourhood_radius = squash_factor * cluster_radius
    alpha_coefficient = 4.0 / (cluster_radius ** 2)
    beta_coefficient = 4.0 / (neighbourhood_radius ** 2)

    candidates = np.zeros(num_samples)
    for i in range(num_samples):
        squared_distances = np.sum((input_data - input_data[i]) ** 2, axis=1)
        candidates[i] = np.sum(np.exp(-alpha_coefficient * squared_distances))

    cluster_centers = []
    initial_candidate = candidates.max()

    while True:
        #  FIND: Pick point with highest candidate as cluster center candidate
        #  DECIDE: Should we accept it?
        #    - Auto-accept if strong enough
        #    - Auto-reject if too weak (break loop)
        #    - Use distance+strength rule if in-between
        best_index = np.argmax(candidates)
        best_candidate_value = candidates[best_index]

        if len(cluster_centers) == 0:
            # Always accept the first center
            cluster_centers.append(input_data[best_index].copy())
            initial_candidate = best_candidate_value
        else:
            ratio_of_candidates = best_candidate_value / initial_candidate

            if ratio_of_candidates > accept_threshold:
                cluster_centers.append(input_data[best_index].copy())
            elif ratio_of_candidates < reject_threshold:
                break
            else:
                # Check shortest distance to existing centers
                # MIDDLE ZONE DECISION: Candidate candidate is neither clearly good nor bad

                # Step 1: Find how far this candidate is from the nearest existing cluster center

                # Step 2: Calculate distance score (0 = right on top, 1 = far away)

                # Step 3: Calculate strength score (0 = weak, 1 = as strong as first cluster)

                # Step 4: Trade-off rule - Accept if sum >= 1.0
                # - Far away + weak = OK (covers new area)
                # - Close + strong = OK (justifies being nearby)
                # - Close + weak = REJECT (redundant and insignificant)

                # If accepted: Add to cluster centers
                # If rejected: Set its candidate to 0 and look for next best candidate
                shortest_distance = min(
                    np.linalg.norm(input_data[best_index] - center) for center in cluster_centers
                )
                if (shortest_distance / cluster_radius) + (best_candidate_value / initial_candidate) >= 1.0:
                    cluster_centers.append(input_data[best_index].copy())
                else:
                    candidates[best_index] = 0.0
                    continue

        new_center = cluster_centers[-1]
        squared_distances = np.sum((input_data - new_center) ** 2, axis=1)
        candidates -= best_candidate_value * \
            np.exp(-beta_coefficient * squared_distances)
        candidates = np.maximum(candidates, 0.0)

    return np.array(cluster_centers)


"""
SUBTRACTIVE CLUSTERING ALGORITHM

PURPOSE: Automatically find cluster centers in data without knowing 
the number of clusters beforehand.

HOW IT WORKS:
    1. Calculate "candidate" for each point (density measure)
    2. Repeatedly:
    - Find point with highest candidate
    - Decide if it's a good cluster center
    - If accepted, SUBTRACT candidate from nearby points
    - Continue until no good candidates remain

WHY "SUBTRACTIVE"? After accepting each center, we reduce (subtract)
the candidate of surrounding points, forcing the algorithm to search
in different regions for the next center.

OUTPUT: Array of cluster centers found, in order of strength.

---------------------------------------
SETUP: Initialize parameters and variables
---------------------------------------
- Number of samples and features are determined from input data.
- Set neighbourhood radius based on squash factor and cluster radius.
- Calculate alpha and beta coefficients for potential management.

---------------------------------------
STEP 1: CALCULATE INITIAL CANDIDATES
---------------------------------------
For each point, calculate how "dense" its neighborhood is. 
Dense areas (many nearby points) = HIGH candidate = good cluster center candidates.
Sparse areas (few nearby points) = LOW candidate = poor candidates. 
Uses Gaussian kernel: nearby points contribute more to potential than distant ones.

---------------------------------------
STEP 2: FIND CLUSTER CENTERS (MAIN LOOP)
---------------------------------------
Iteratively find cluster centers until no good candidates remain.
Each iteration:
1. FIND: Pick point with highest remaining candidate
2. DECIDE: Accept/reject based on strength and distance from existing centers
3. SUBTRACT: Reduce candidate around accepted center (the "subtractive" part)
4. REPEAT: Loop continues until candidates too weak

DECISION LOGIC: Should we accept this candidate as a cluster center?
THREE CASES:
1. FIRST CENTER: Always accept (no comparison needed)
2. STRONG CANDIDATE: Auto-accept if candidate > accept_threshold
3. WEAK CANDIDATE: Auto-reject if candidate < reject_threshold (STOP)
4. MIDDLE ZONE: Use distance+strength trade-off rule

---------------------------------------
STEP 3: SUBTRACT CANDIDATES (Suppression)
---------------------------------------
After accepting a cluster center, reduce the candidate of all nearby points.
WHY? Prevents selecting another center too close to this one.
Creates "exclusion zone" around accepted centers.
Forces next iteration to look in different regions.
How much to subtract: Points CLOSE to center → lose lots of candidate (heavily suppressed),
Points FAR from center → barely affected, using Gaussian decay with beta_coefficient.

---------------------------------------
RETURN: Array of discovered cluster centers
---------------------------------------
Centers are in order of strength (first = strongest cluster).
"""
