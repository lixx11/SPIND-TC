#!/usr/bin/env python

"""
Usage:
    batch_index_sim.py <PEAK-FILE> <TABLE-FILE> <PHOTON-ENERGY-LIST> <DET-DIST> <PIXEL-SIZE> [options]

Options:
    -h --help                   Show this screen.
    --output=<file>             Specify output file [default: spind.csv].
    --max-index=<num>           Max number of patterns to index [default: -1].
    --seed-len-tol=<num>        Specify seed length tolerance in per angstrom[default: 0.001].
    --seed-angle-tol=<num>      Specify seed angle tolerance in degree [default: 1.].
    --seed-pair-num=<num>       Specify maximum number of seed pairs [default: 100].
    --eval-tol=<float>          Specify evalulation tolerance [default: 0.25].
"""

try:
    import mkl
    mkl.set_num_threads(1)  # disable numpy multi-thread parallel computation
except:
    pass
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from docopt import docopt
import sys
import time

from util import load_table, calc_transform_matrix
from geometry import Detector
from index import index
from util import rotation_matrix_to_euler_angles


if __name__ == "__main__":
    # parse options
    args = docopt(__doc__)
    peak_file = args['<PEAK-FILE>']
    table_file = args['<TABLE-FILE>']
    photon_energy_list = list(map(float, args['<PHOTON-ENERGY-LIST>'].split(',')))
    det_dist = float(args['<DET-DIST>'])
    pixel_size = float(args['<PIXEL-SIZE>'])
    max_index = int(args['--max-index'])
    seed_len_tol = float(args['--seed-len-tol'])
    seed_angle_tol = float(args['--seed-angle-tol'])
    seed_pair_num = int(args['--seed-pair-num'])
    eval_tol = float(args['--eval-tol'])

    peak_data = np.load(peak_file)
    if max_index != -1:
        peak_data = peak_data[:max_index]
    table = load_table(table_file)
    table['A0'] = calc_transform_matrix(table['lattice_constants'])
    fout = open(args['--output'], 'w')
    fout.write(
        'id,nb_peak,time,nb_seed_pair,nb_solution,match_rate,pair_dist,'
        't1,t2,t3,'
        'tp1,tp2,tp3,'
        'd1,d2,d3\n'
    )

    for i in tqdm(range(len(peak_data))):
        t0 = time.time()
        coords = peak_data[i]['coords'][:, ::-1]

        res = index(
            table, coords, photon_energy_list, det_dist, 
            seed_len_tol=seed_len_tol,
            seed_angle_tol=seed_angle_tol,
            seed_pair_num=seed_pair_num,
            eval_tol=eval_tol,
            match_rate_thres=0.99,
            sort='resolution',
            refine=False,
            show_progress=False,
        )
        t1 = time.time()

        solution = res['best_solution']
        euler_angles = np.array(rotation_matrix_to_euler_angles(solution.R))
        true_euler_angles = peak_data[i]['euler_angles']
        diff_euler_angles = np.abs(euler_angles - true_euler_angles)
        print('=' * 100)
        print('time elapsed %.2f sec' % (t1 - t0))
        print('searched seed pairs: %d' % res['seed_pair_count'])
        print('searched solution candidates: %d' % res['solution_count'])
        print('match rate: %.3f' % solution.match_rate)
        print('pair_dist: %.3e' % solution.pair_dist)
        print('best solution: ', euler_angles)
        print('true solution: ', true_euler_angles)
        print('best solution: ', solution.R)
        print('true solution: ', peak_data[i]['rotation_matrix'])
        print('=' * 100)

        # probs = res['probs_c1']
        # true_photon_energy = peak_data[i]['photon_energy']

        # fig, ax = plt.subplots(figsize=(8, 4))
        # idx = np.where(true_photon_energy == 9000)[0]
        # ax.scatter(idx, probs[idx], c='C0')
        # idx = np.where(true_photon_energy == 7000)[0]
        # ax.scatter(idx, probs[idx], c='C1')
        # ax.set_ylim(0, 1.0)
        # ax.set_ylabel('probability of first color')
        # ax.set_xlabel('index')
        # ax.axhline(y=0.5, linestyle='--', color='red')
        # plt.tight_layout()
        # plt.show()

        fout.write(
            '%d,%d,%.2f,%d,%d,'
            '%.3f,%.3e,'
            '%.2f,%.2f,%.2f,'
            '%.2f,%.2f,%.2f,'
            '%.2f,%.2f,%.2f\n' % (
                i, coords.shape[0], t1 - t0, res['seed_pair_count'], res['solution_count'],
                solution.match_rate, solution.pair_dist,
                euler_angles[0], euler_angles[1], euler_angles[2],
                true_euler_angles[0], true_euler_angles[1], true_euler_angles[2],
                diff_euler_angles[1], diff_euler_angles[1], diff_euler_angles[2]
            )
        )
        fout.flush()
