#!/usr/bin/env python

import sys
import numpy as np
from numpy.linalg import norm, inv
from scipy.optimize import minimize
from itertools import combinations
from sklearn.cluster import DBSCAN
from util import det2fourier, calc_wavelength, calc_angle, load_table, \
    calc_transform_matrix, calc_rotation_matrix, rotation_matrix_to_euler_angles
from tqdm import tqdm
import time
import matplotlib.pyplot as plt


class Solution(object):
    def __init__(self, A):
        self.A = A
        self.match_rate = 0.


def build_seed_pool(prob, weight=None, size=5):
    """
    Build seed pool for indexing.
    
    prob: 1d array of probabilities for COI (color of interested).
    weight: 1d array of extra weight.
    size: size of seed pool.
    
    Return: indices of seed pool.
    """

    priorities = prob.copy()
    if weight is not None:
        priorities *= weight
    seed_pool = np.argsort(priorities)[-size:]
    return seed_pool


def search_table(q_pair, table, seed_len_tol=0.001, seed_angle_tol=1.):
    """
    Search solution candidates.
    
    q_pair: 2x3 array (q1, q2).
    table: reference table.
    
    return: solution cadidates, q1, q2
    """
    q1, q2 = q_pair.copy()
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
    return candidates, q1, q2


def build_solution(A0, q1, q2, hkl1, hkl2):
    """
    Build indexing solution.
    
    A0: original transform matrix.
    q1, q2: experimental q pair.
    hkl1, hkl2: matched hkl pair.
    
    return: indexing solution.
    """
    ref_q1 = A0.dot(hkl1)
    ref_q2 = A0.dot(hkl2)
    R = calc_rotation_matrix(q1, q2, ref_q1, ref_q2)
    A = R.dot(A0)
    solution = Solution(A)
    solution.R = R
    return solution


def eval_solution(solution, qs_c1, qs_c2, calc_pair_dist=False):
    """
    Evaluate solution.
    
    qs_c1: Nx3 array, q vectors of first color.
    qs_c2: Nx3 array, q vectors of second color.
    """
    hkls_c1 = np.linalg.inv(solution.A).dot(qs_c1.T).T
    rhkls_c1 = np.round(hkls_c1)
    ehkls_c1 = np.abs(hkls_c1 - rhkls_c1)

    hkls_c2 = np.linalg.inv(solution.A).dot(qs_c2.T).T
    rhkls_c2 = np.round(hkls_c2)
    ehkls_c2 = np.abs(hkls_c2 - rhkls_c2)

    match_rate = np.mean((np.max(ehkls_c1, axis=1) <= 0.25) | (np.max(ehkls_c2, axis=1) <= 0.25))
    solution.match_rate = match_rate

    if calc_pair_dist:
        probs_c1, probs_c2 = calc_prob(solution, qs_c1, qs_c2)
        pair_ids_c1 = np.where(
            (np.max(ehkls_c1, axis=1) < 0.25) * (probs_c1 > 0.5)
        )[0]
        pair_ids_c2 = np.where(
            (np.max(ehkls_c2, axis=1) < 0.25) * (probs_c2 > 0.5)
        )[0]
        eXYZs_c1 = solution.A.dot(rhkls_c1.T).T - qs_c1
        eXYZs_c2 = solution.A.dot(rhkls_c2.T).T - qs_c2
        pair_dist = (np.sum(norm(eXYZs_c1, axis=1)[pair_ids_c1]) + np.sum(norm(eXYZs_c2, axis=1)[pair_ids_c2])) \
            / (len(pair_ids_c1) + len(pair_ids_c2))
        solution.pair_dist = pair_dist
        solution.hkls_c1 = hkls_c1
        solution.rhkls_c1 = rhkls_c1
        solution.ehkls_c1 = ehkls_c1
        solution.hkls_c2 = hkls_c2
        solution.rhkls_c2 = rhkls_c2
        solution.ehkls_c2 = ehkls_c2


def calc_prob(solution, qs_c1, qs_c2):
    rhkls1 = np.round(np.linalg.inv(solution.A).dot(qs_c1.T).T)
    rhkls2 = np.round(np.linalg.inv(solution.A).dot(qs_c2.T).T)
    residue1 = norm(solution.A.dot(rhkls1.T).T - qs_c1, axis=1)
    residue2 = norm(solution.A.dot(rhkls2.T).T - qs_c2, axis=1)
    probs_c1 = (1./residue1) / (1./residue1 + 1./residue2)
    probs_c2 = 1 - probs_c1
    return probs_c1, probs_c2


def cluster_solutions(solutions, eps=1.0, min_samples=1):
    """
    Cluster solution by 3 Euler angles.
    
    return: solution clusters, sorted by match rate of best element.
    """
    euler_angles = np.array([
        rotation_matrix_to_euler_angles(solution.R) for solution in solutions
    ])
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(euler_angles)
    labels = db.labels_
    cluster_labels = np.unique(db.labels_)
    clusters = []
    for cluster_label in cluster_labels:
        elements = [solutions[i]
                    for i in range(len(solutions)) if labels[i] == cluster_label]
        match_rates = [element.match_rate for element in elements]
        clusters.append({
            'label': cluster_label,
            'el': elements,
            'best_el': elements[np.argmax(match_rates)]
        })
    clusters.sort(key=lambda x: x['best_el'].match_rate, reverse=True)
    return clusters


def refine_solution(solution, qs_c1, qs_c2, probs_c1, probs_c2, min_prob=0.5):
    def func(x):
        asx, bsx, csx, asy, bsy, csy, asz, bsz, csz = x
        A = np.array([
            [asx, bsx, csx],
            [asy, bsy, csy],
            [asz, bsz, csz]
        ])
        eXYZs_c1 = A.dot(solution.rhkls_c1.T).T - qs_c1
        eXYZs_c2 = A.dot(solution.rhkls_c2.T).T - qs_c2
        pair_ids_c1 = np.where(
            (np.max(solution.ehkls_c1, axis=1) < 0.25) * (probs_c1 > min_prob)
        )[0]
        pair_ids_c2 = np.where(
            (np.max(solution.ehkls_c2, axis=1) < 0.25) * (probs_c2 > min_prob)
        )[0]
        pair_dist = (
            np.sum(norm(eXYZs_c1, axis=1)[pair_ids_c1]) +
            np.sum(norm(eXYZs_c2, axis=1)[pair_ids_c2])
        ) / (len(pair_ids_c1) + len(pair_ids_c2))

        return pair_dist

    res = minimize(func, solution.A.reshape(-1), options={'disp': False})
    A_refined = res.x.reshape(3, 3)
    pair_dist_refined = res.fun

    # calculate refined match rate
    hkls_c1 = np.linalg.inv(A_refined).dot(qs_c1.T).T
    rhkls_c1 = np.round(hkls_c1)
    ehkls_c1 = np.abs(hkls_c1 - rhkls_c1)
    hkls_c2 = np.linalg.inv(A_refined).dot(qs_c2.T).T
    rhkls_c2 = np.round(hkls_c2)
    ehkls_c2 = np.abs(hkls_c2 - rhkls_c2)
    match_rate_refined = np.mean(
        (np.max(ehkls_c1, axis=1) <= 0.25) | (np.max(ehkls_c2, axis=1) <= 0.25))

    # evaluate two colors
    peaks_c1 = probs_c1 >= 0.5
    match_peaks_c1 = (np.max(ehkls_c1[peaks_c1], axis=1) <= 0.25).sum()
    match_rate_c1 = match_peaks_c1 / float(peaks_c1.sum()) 
    peaks_c2 = probs_c2 >= 0.5
    match_peaks_c2 = (np.max(ehkls_c2[peaks_c2], axis=1) <= 0.25).sum()
    match_rate_c2 = match_peaks_c2 / float(peaks_c2.sum())

    if pair_dist_refined > solution.pair_dist:  # keep original solution
        solution.A_refined = solution.A
        solution.pair_dist_refined = solution.pair_dist
        solution.match_rate_refined = solution.match_rate
    else:
        solution.A_refined = A_refined
        solution.pair_dist_refined = pair_dist_refined
        solution.match_rate_refined = match_rate_refined
    solution.peaks_c1 = peaks_c1.sum()
    solution.match_rate_c1 = match_rate_c1
    solution.peaks_c2 = peaks_c2.sum()
    solution.match_rate_c2 = match_rate_c2


def index(
    table, peaks, photon_energy_list, det_dist,
    seed_len_tol=0.001,
    seed_angle_tol=1.,
    seed_pair_num=100,
    refine=True,
    show_progress=True,
    sort=None,
    intensity=None,
    snrs=None,
    pre_solution=None,
):
    if show_progress:
        tqdm_ = tqdm
    else:
        def tqdm_(x): return x
    high_res = table['high_res']
    A0 = table['A0']

    # initiation for all colors
    wavelength = []
    peak_rvec = []
    peak_res = []
    peak_probs = []
    for photon_energy in photon_energy_list:
        wavelength_ = calc_wavelength(photon_energy)
        peak_rvec_ = det2fourier(peaks, wavelength_, det_dist) / 1E10
        peak_res_ = 1. / norm(peak_rvec_, axis=1)
        peak_prob_ = np.zeros_like(peak_res_)
        wavelength.append(wavelength_)
        peak_rvec.append(peak_rvec_)
        peak_res.append(peak_res_)
        peak_probs.append(peak_prob_)

    # do searching 
    solutions = []
    best_match_rate = 0.
    seed_pair_count = 0
    solution_count = 0
    for i in range(len(photon_energy_list)):
        seed_pool = np.where(peak_res[i] > high_res)[0]
        seed_pairs = list(combinations(seed_pool, 2))
        # sort peak pairs
        if sort == 'intensity' and intensity is not None:
            seed_pairs.sort(
                key=lambda p: intensity[p[0]] + intensity[p[1]], reverse=True
            )
        elif sort == 'resolution':
            seed_pairs.sort(
                key=lambda p: peak_res[i][p[0]] + peak_res[i][p[1]], reverse=False
            )
        elif sort == 'snr' and snrs is not None:
            seed_pairs.sort(
                key=lambda p: snrs[p[0]] + snrs[p[1]], reverse=True
            )
        # do heavy searching
        for seed_pair in tqdm_(seed_pairs[:seed_pair_num]):
            candidates, q1, q2 = search_table(
                peak_rvec[i][seed_pair, :], table, seed_len_tol=seed_len_tol, seed_angle_tol=seed_angle_tol
            )
            for candidate in candidates:
                solution = build_solution(
                    A0, q1, q2,
                    table['hkl1'][candidate],
                    table['hkl2'][candidate]
                )
                if pre_solution is not None:
                    solution.A = pre_solution
                eval_solution(solution, peak_rvec[0], peak_rvec[1])
                solutions.append(solution)
                best_match_rate = max(best_match_rate, solution.match_rate)
                solution_count += 1
            seed_pair_count += 1
            if best_match_rate > 0.7:
                break
        if best_match_rate > 0.7:
            break

    solutions.sort(key=lambda s: s.match_rate, reverse=True)
    clusters = cluster_solutions(solutions[:1000])
    # print('cluster/solution num: %d/%d' % (len(clusters), len(solutions)))

    best_match_rate = clusters[0]['best_el'].match_rate
    # print('best match rate: %.3f' % best_match_rate)
    top_solutions = [cluster['best_el']
                     for cluster in clusters if cluster['best_el'].match_rate == best_match_rate]

    for solution in top_solutions:
        eval_solution(solution, peak_rvec[0], peak_rvec[1], calc_pair_dist=True)
    top_solutions.sort(key=lambda s: s.pair_dist, reverse=True)
    best_solution = top_solutions[0]

    probs_c1, probs_c2 = calc_prob(best_solution, peak_rvec[0], peak_rvec[1])

    if refine:
        refine_solution(
            best_solution, peak_rvec[0], peak_rvec[1], probs_c1, probs_c2, min_prob=0.5)
    
    del best_solution.hkls_c1
    del best_solution.rhkls_c1
    del best_solution.ehkls_c1
    del best_solution.hkls_c2
    del best_solution.rhkls_c2
    del best_solution.ehkls_c2
    
    res_dict = {
        'best_solution': best_solution,
        'probs_c1': probs_c1,
        'probs_c2': probs_c2,
        'solution_count': solution_count,
        'seed_pair_count': seed_pair_count,
    }
    return res_dict


if __name__ == "__main__":
    photon_energy_list = [7007.873, 8949.838]
    det_dist = 0.05103
    pixel_size = 50E-6
    center = np.array([1056.363, 1080.320])

    peaks_file = 'data/exp1/peaks_TC.txt'
    table_file = 'data/exp1/table3.h5'
    peaks = np.loadtxt(peaks_file)
    peaks = peaks - center
    coords = peaks * pixel_size

    table = load_table(table_file)
    table['A0'] = calc_transform_matrix(table['lattice_constants'])

    t0 = time.time()
    res = index(table, coords, photon_energy_list, det_dist)
    t1 = time.time()

    plt.hist(res['probs_c1'], bins=np.arange(0, 1, 0.05))
    plt.show()

    print('=' * 100)
    print('time elapsed: %.3f' % (t1 - t0))
    print('searched seed pairs: %d' % res['seed_pair_count'])
    print('searched solution candidates: %d' % res['solution_count'])
    print('total peaks: %d' % coords.shape[0])
    print('match rate(before refine): %.3f' % res['best_solution'].match_rate)
    print('match rate(after refine): %.3f' % res['best_solution'].match_rate_refined)
    print('pair_dist(before refine): %.3e' % res['best_solution'].pair_dist)
    print('pair_dist(after refine): %.3e' % res['best_solution'].pair_dist_refined)
    print('c1 peaks: %d' % res['best_solution'].peaks_c1)
    print('c1 match rate: %.3f' % res['best_solution'].match_rate_c1)
    print('c2 peaks: %d' % res['best_solution'].peaks_c2)
    print('c2 match rate: %.3f' % res['best_solution'].match_rate_c2)
