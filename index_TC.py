#!/usr/bin/env python

import sys
import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize
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
TOP_SIZE = 3


class Solution(object):
    pass


if __name__ == "__main__":
    center = np.array([CENTER_FS, CENTER_SS])
    photon_energy1 = 9000
    photon_energy2 = 7000
    wavelength1 = calc_wavelength(photon_energy1)
    wavelength2 = calc_wavelength(photon_energy2)
    det_dist = 0.1
    seed_len_tol = 0.001
    seed_angle_tol = 1.
    peaks_file = '5oer/peaks_TC.txt'
    table_file = '5oer/table.h5'

    # process peaks
    peaks = np.loadtxt(peaks_file)
    peaks = peaks - center
    coords = peaks * PIXEL_SIZE
    coords = -coords  # to be consistent with simulator setup
    q1s = det2fourier(coords, wavelength1, det_dist) * \
        1E-10  # q vectors of color 1
    q2s = det2fourier(coords, wavelength2, det_dist) * \
        1E-10  # q vectors of color 2
    p1s = np.random.rand(q1s.shape[0])  # probabilities of color 1
    p2s = 1 - p1s  # probabilities of color 2

    # process hkl table
    table = load_table(table_file)
    A0 = calc_transform_matrix(table['lattice_constants'])

    for iter in range(10):
        print('iter %d' % iter)
        seed_pool = sorted(list(range(q1s.shape[0])), key=lambda i: p1s[int(i)], reverse=True)[:5]
        print(seed_pool)
        seed_pairs = list(combinations(seed_pool, 2))
        solutions = []
        for seed_pair in seed_pairs:
            q1, q2 = q1s[seed_pair, :]
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
                # evaluate this solution for color 1
                _hkls1 = np.linalg.inv(A).dot(q1s.T).T
                _rhkls1 = np.round(_hkls1)
                _ehkls1 = np.abs(_hkls1 - _rhkls1)
                _paired_ids1 = np.where(np.max(_ehkls1, axis=1) < 0.25)[0]
                _match_rate1 = float(_paired_ids1.size) / float(peaks.shape[0])
                # evaluate this solution for color 2
                _hkls2 = np.linalg.inv(A).dot(q2s.T).T
                _rhkls2 = np.round(_hkls2)
                _ehkls2 = np.abs(_hkls2 - _rhkls2)
                _paired_ids2 = np.where(np.max(_ehkls2, axis=1) < 0.25)[0]
                _match_rate2 = float(_paired_ids2.size) / float(peaks.shape[0])
                # weighted match rate
                match_rate = np.mean(
                    (np.max(_ehkls1, axis=1) < 0.25).astype(int) * p1s 
                    + (np.max(_ehkls2, axis=1) < 0.25).astype(int) * p2s
                )
                solution = Solution()
                solution.R = R
                solution.match_rate1 = _match_rate1
                solution.match_rate2 = _match_rate2
                solution.match_rate = match_rate
                solution.seed_pair = seed_pair
                solution.hkl1 = hkl1
                solution.hkl2 = hkl2
                solution.candidate = candidate
                solution.len_angle = (q1_len, q2_len, angle) 
                solutions.append(solution)
        print('solution num: %d' % len(solutions))
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
        print('cluster num: %d' % len(clusters))
        # pp = pprint.PrettyPrinter(indent=4)
        # pp.pprint(clusters)

        fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
        cm = plt.cm.get_cmap('RdYlBu')
        sc = ax[0].scatter(
            euler_angles[:, 0],
            euler_angles[:, 1],
            s=30,
            vmin=0., vmax=1., cmap=cm,
            c=[s.match_rate for s in solutions]
        )
        ax[0].set(xlim=(-180,180), ylim=(-90, 90))
        ax[0].set_title('solutions')
        plt.colorbar(sc)
        ax[1].plot(p1s)
        ax[1].set_title('probability of color 1')
        ax[1].set_ylim(0, 1)
        plt.suptitle('iter %d' % iter)
        plt.savefig('fig%d.png' % iter)
        plt.close()
        # plt.show()

        top_solutions = [solutions[cluster['best_el']] for cluster in clusters[:TOP_SIZE]]
        top_clusters = clusters[:TOP_SIZE]
        for cluster in top_clusters:
            solution = solutions[cluster['best_el']]
            print('number of el: %d' % len(cluster['el']))
            print(solution.match_rate, rotation_matrix_to_euler_angles(solution.R))# , solution.seed_pair, solution.hkl1, solution.hkl2, solution.candidate, solution.len_angle)
        
        best_solution = top_solutions[0]
        A = best_solution.R.dot(A0)
        rhkls1 = np.round(np.linalg.inv(A).dot(q1s.T).T)
        rhkls2 = np.round(np.linalg.inv(A).dot(q2s.T).T)
        residue1 = norm(A.dot(rhkls1.T).T - q1s, axis=1)
        residue2 = norm(A.dot(rhkls2.T).T - q2s, axis=1)
        p1s = (1./residue1) / (1./residue1 + 1./residue2)
        p2s = 1 - p1s
