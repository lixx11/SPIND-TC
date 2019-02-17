#!/usr/bin/env python

import sys
import numpy as np
from numpy.linalg import norm
from itertools import combinations
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pprint
from util import det2fourier, calc_wavelength, calc_angle, load_table, \
    calc_transform_matrix, calc_rotation_matrix, rotation_matrix_to_euler_angles


FS = np.array([-1., 0., 0.])  # detector fast scan vector
SS = np.array([0., -1., 0.])  # detector slow scan vector
PIXEL_SIZE = 100E-6  # pixel size in meter
CENTER_FS = 720  # beam center alone fast scan direction
CENTER_SS = 720  # beam center alone slow scan direction
TOP_SIZE = 10


class Solution(object):
    pass


if __name__ == "__main__":
    center = np.array([CENTER_FS, CENTER_SS])
    photon_energy = 9000
    wavelength = calc_wavelength(photon_energy)
    det_dist = 0.1
    seed_len_tol = 0.001
    seed_angle_tol = 1.
    peaks_file = 'tmp/peaks.txt'
    table_file = 'tmp/table.h5'

    # process peaks
    peaks = np.loadtxt(peaks_file)
    peaks = peaks - center
    coords = peaks * PIXEL_SIZE
    coords = -coords  # to be consistent with simulator setup
    qs = det2fourier(coords, wavelength, det_dist) * 1E-10  # convert to per angstrom

    # process hkl table
    table = load_table(table_file)
    A0 = calc_transform_matrix(table['lattice_constants'])

    seed_pools = list(range(qs.shape[0]))[:5]
    seed_pairs = list(combinations(seed_pools, 2))
    solutions = []
    for seed_pair in seed_pairs:
        q1, q2 = qs[seed_pair, :]
        q1_len, q2_len = norm(q1), norm(q2)
        if q1_len < q2_len:
            q1, q2 = q2, q1
            q1_len, q2_len = q2_len, q1_len
        angle = calc_angle(q1, q2, q1_len, q2_len)
        candidates = np.where(
            (np.abs(q1_len - table['len_angle'][:, 0]) < seed_len_tol)
            * (np.abs(q2_len - table['len_angle'][:, 1]) < seed_len_tol)
            * (np.abs(angle - table['len_angle'][:, 2]) < seed_angle_tol)
        )[0]

        for candidate in candidates:
            hkl1 = table['hkl1'][candidate]
            hkl2 = table['hkl2'][candidate]
            ref_q1 = A0.dot(hkl1)
            ref_q2 = A0.dot(hkl2)
            R = calc_rotation_matrix(q1, q2, ref_q1, ref_q2)
            A = R.dot(A0)
            # evaluate this solution
            _hkls = np.linalg.inv(A).dot(qs.T).T
            _rhkls = np.round(_hkls)
            _ehkls = np.abs(_hkls - _rhkls)
            _paired_ids = np.where(np.max(_ehkls, axis=1) < 0.25)[0]
            _match_rate = float(_paired_ids.size) / float(peaks.shape[0])
            solution = Solution()
            solution.match_rate = _match_rate
            solution.R = R
            solutions.append(solution)
    # do clustering
    euler_angles = np.array([
        rotation_matrix_to_euler_angles(solution.R) for solution in solutions
    ])
    db = DBSCAN(eps=1.0, min_samples=1).fit(euler_angles)
    labels = db.labels_
    cluster_labels = np.unique(db.labels_)
    clusters = []
    for cluster_label in cluster_labels:
        elements = np.where(db.labels_ == cluster_label)[0]
        match_rates = [solutions[i].match_rate for i in elements]
        clusters.append({
            'label': cluster_label,
            'el': elements,
            'match_rates': match_rates,
            'best_match_rate': max(match_rates),
            'best_el': elements[np.argmax(match_rates)]
        })
    clusters.sort(key=lambda x: x['best_match_rate'], reverse=True)
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(clusters)

    # fig, ax = plt.subplots()
    # cm = plt.cm.get_cmap('RdYlBu')
    # for cluster in clusters:
    #     sc = ax.scatter(
    #         euler_angles[cluster['el']][:, 0], 
    #         euler_angles[cluster['el']][:, 1], 
    #         s=30,
    #         vmin=0., vmax=1., cmap=cm,
    #         c=cluster['match_rates'])
    # plt.colorbar(sc)
    # plt.show()

    top_solutions = [solutions[cluster['best_el']] for cluster in clusters[:TOP_SIZE]]
    for solution in top_solutions:
        print(solution.match_rate, rotation_matrix_to_euler_angles(solution.R))
