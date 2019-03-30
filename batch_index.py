#!/usr/bin/env python

"""
Usage:
    batch_index.py <PEAK-FILE> <GEOM-FILE> <TABLE-FILE> <PHOTON-ENERGY-LIST> <DET-DIST> <PIXEL-SIZE> [options]

Options:
    -h --help                   Show this screen.
    --output=<file>             Specify output file [default: spind.csv].
    --sort-by=<method>          Sort peaks by intensity, snr or resolution [default: none].
    --presolution=<solution>    Use presolution in peak file [default: none].
    --seed-len-tol=<num>        Specify seed length tolerance in per angstrom[default: 0.001].
    --seed-angle-tol=<num>      Specify seed angle tolerance in degree [default: 1.].
    --seed-pair-num=<num>       Specify maximum number of seed pairs [default: 100].
    --refine=<refine>           Whether to refine final solution [default: true].
    --show-progress=<progress>  Whether to show indexing progress [default: true].
"""

try:
    import mkl
    mkl.set_num_threads(1)  # disable numpy multi-thread parallel computation
except:
    pass
import numpy as np

from docopt import docopt
import sys
import time

from util import load_table, calc_transform_matrix
from geometry import Detector
from index import index


if __name__ == "__main__":
    # parse options
    args = docopt(__doc__)
    peak_file = args['<PEAK-FILE>']
    geom_file = args['<GEOM-FILE>']
    table_file = args['<TABLE-FILE>']
    photon_energy_list = list(map(float, args['<PHOTON-ENERGY-LIST>'].split(',')))
    det_dist = float(args['<DET-DIST>'])
    pixel_size = float(args['<PIXEL-SIZE>'])
    sort_by = args['--sort-by']
    pre_solution_path = args['--presolution']
    refine = args['--refine']
    show_progress = args['--show-progress']
    seed_len_tol = float(args['--seed-len-tol'])
    seed_angle_tol = float(args['--seed-angle-tol'])
    seed_pair_num = int(args['--seed-pair-num'])
    sort_by = None if sort_by == 'none' else sort_by
    pre_solution_path = None if pre_solution_path == 'none' else pre_solution_path
    refine = True if refine == 'true' else False 
    show_progress = True if show_progress == 'true' else False

    peak_data = np.load(peak_file)
    table = load_table(table_file)
    table['A0'] = calc_transform_matrix(table['lattice_constants'])
    detector = Detector(geom_file, ['q%d' % i for i in range(1, 9)])
    fout = open(args['--output'], 'w')
    fout.write(
        'id,nb_peak,time,nb_seed_pair,nb_solution,match_rate,pair_dist,'
        'nb_peak_c1,match_rate_c1,nb_peak_c2,match_rate_c2,'
        'ax,ay,az,bx,by,bz,cx,cy,cz\n'
    )

    for i in range(len(peak_data[:10])):
        t0 = time.time()
        image_file = peak_data[i]['image_file']
        peaks = peak_data[i]['peaks']
        pre_solution = peak_data[i][pre_solution_path] if pre_solution_path is not None else None

        coords_x, coords_y = detector.map2xy(peaks[:, 0], peaks[:, 1])
        coords = np.vstack([coords_x, coords_y]).T
        coords *= pixel_size
        intensity = peaks[:, 2]

        res = index(
            table, coords, photon_energy_list, det_dist, 
            seed_len_tol=seed_len_tol,
            seed_angle_tol=seed_angle_tol,
            seed_pair_num=seed_pair_num,
            intensity=intensity,
            pre_solution=pre_solution,
            sort=sort_by,
            refine=refine,
            show_progress=show_progress,
        )
        t1 = time.time()

        solution = res['best_solution']
        A = solution.A_refined
        print('=' * 100)
        print('time elapsed %.2f sec' % (t1 - t0))
        print('searched seed pairs: %d' % res['seed_pair_count'])
        print('searched solution candidates: %d' % res['solution_count'])   
        print('match rate(refine refine): %.3f' % solution.match_rate)
        print('match rate(after refine): %.3f' % solution.match_rate_refined)
        print('pair_dist(before refine): %.3e' % solution.pair_dist)
        print('pair_dist(after refine): %.3e' % solution.pair_dist_refined)
        print('best solution: ', solution.A)
        print('=' * 100)

        fout.write(
            '%d,%d,%.2f,%d,%d,'
            '%.3f,%.3e,'
            '%d,%.3f,%d,%.3f,'
            '%.4e,%.4e,%.4e,'
            '%.4e,%.4e,%.4e,'
            '%.4e,%.4e,%.4e\n' % (
                i, coords.shape[0], t1 - t0, res['seed_pair_count'], res['solution_count'],
                solution.match_rate_refined, solution.pair_dist_refined,
                solution.peaks_c1, solution.match_rate_c1, solution.peaks_c2, solution.match_rate_c2,
                A[0, 0], A[1, 0], A[2, 0],  # a*
                A[0, 1], A[1, 1], A[2, 1],  # b*
                A[0, 2], A[1, 2], A[2, 2],  # c*
            )
        )
        fout.flush()
