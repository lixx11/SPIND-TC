#!/usr/bin/env python

"""
Usage:
    batch_index_mpi.py <PEAK-FILE> <GEOM-FILE> <TABLE-FILE> <PHOTON-ENERGY-LIST> <DET-DIST> <PIXEL-SIZE> [options]

Options:
    -h --help                   Show this screen.
    --output=<file>             Specify output file [default: spind.csv].
    --max-index=<num>          Max number of patterns to index [default: -1].
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

from mpi4py import MPI
import numpy as np

from docopt import docopt
import sys
import time

from util import load_table, calc_transform_matrix
from geometry import Detector
from index import index


def build_index_jobs(peak_data, detector, pixel_size, pre_solution_path=None):
    index_job = []
    for i in range(len(peak_data)):
        image_file = peak_data[i]['image_file']
        peaks = peak_data[i]['peaks']
        pre_solution = peak_data[i][pre_solution_path] if pre_solution_path is not None else None

        coords_x, coords_y = detector.map2xy(peaks[:, 0], peaks[:, 1])
        coords = np.vstack([coords_x, coords_y]).T
        coords *= pixel_size
        intensity = peaks[:, 2] if peaks.shape[1] >= 3 else None
        snr = peaks[:, 3] if peaks.shape[1] >= 4 else None

        peak_dict = {
            'id': i,
            'image_file': image_file,
            'coords': coords,
            'pre_solution': pre_solution,
            'intensity': intensity,
            'snr': snr,
        }
        index_job.append(peak_dict)
    return index_job


def write_to_csv(res, fout):
    if res is None:
        return
    solution = res['best_solution']
    A = solution.A_refined
    fout.write(
        '%d,%d,%.2f,%d,%d,'
        '%.3f,%.3e,'
        '%d,%.3f,%d,%.3f,'
        '%.4e,%.4e,%.4e,'
        '%.4e,%.4e,%.4e,'
        '%.4e,%.4e,%.4e\n' % (
            res['id'], res['nb_peak'], res['time'],
            res['seed_pair_count'], res['solution_count'],
            solution.match_rate_refined, solution.pair_dist_refined,
            solution.peaks_c1, solution.match_rate_c1, solution.peaks_c2, solution.match_rate_c2,
            A[0, 0], A[1, 0], A[2, 0],  # a*
            A[0, 1], A[1, 1], A[2, 1],  # b*
            A[0, 2], A[1, 2], A[2, 2],  # c*
        )
    )
    fout.flush()


def master_run(args):
    print('master running')
    # buffer = bytearray(1<<18)
    peak_file = args['<PEAK-FILE>']
    geom_file = args['<GEOM-FILE>']
    pixel_size = float(args['<PIXEL-SIZE>'])
    max_index = int(args['--max-index'])
    pre_solution_path = args['--presolution']
    pre_solution_path = None if pre_solution_path == 'none' else pre_solution_path

    peak_data = np.load(peak_file)
    if max_index != -1:
        peak_data = peak_data[:max_index]
    detector = Detector(geom_file, ['q%d' % i for i in range(1, 9)])

    fout = open(args['--output'], 'w')
    fout.write(
        'id,nb_peak,time,nb_seed_pair,nb_solution,match_rate,pair_dist,'
        'nb_peak_c1,match_rate_c1,nb_peak_c2,match_rate_c2,'
        'ax,ay,az,bx,by,bz,cx,cy,cz\n'
    )

    # distribute jobs
    jobs = build_index_jobs(peak_data, detector, pixel_size, pre_solution_path=pre_solution_path)
    job_num = len(jobs)
    job_id = 0
    reqs = {}
    workers = set(range(1, size))
    for worker in workers:
        if job_id < job_num:
            job = jobs[job_id]
        else:
            job = None  # dummy job
        comm.isend(job, dest=worker)
        print('%d/%d  --> %d' % (job_id, job_num, worker), flush=True)
        reqs[worker] = comm.irecv(source=worker)
        job_id += 1
    
    finished_workers = set()
    while job_id < job_num:
        stop = False
        time.sleep(0.001)  # take a break
        workers -= finished_workers
        for worker in workers:
            finished, res = reqs[worker].test()
            if finished:
                write_to_csv(res, fout)
                if job_id < job_num:
                    comm.isend(stop, dest=worker)
                    comm.isend(jobs[job_id], dest=worker)
                    print('%d/%d --> %d' %
                          (job_id, job_num, worker), flush=True)
                    reqs[worker] = comm.irecv(source=worker)
                    job_id += 1
                else:
                    stop = True
                    comm.isend(stop, dest=worker)
                    finished_workers.add(worker)
    
    all_processed = False
    while not all_processed:
        time.sleep(0.001)
        all_processed = True
        workers -= finished_workers
        for worker in workers:
            finished, res = reqs[worker].test()
            if finished:
                write_to_csv(res, fout)
                stop = True
                comm.isend(stop, dest=worker)
                finished_workers.add(worker)
            else:
                all_processed = False
    
    fout.close()
    print('Done!')


def worker_run(args):
    print('worker(%d) running' % rank)
    det_dist = float(args['<DET-DIST>'])
    sort_by = args['--sort-by']
    refine = args['--refine']
    show_progress = args['--show-progress']
    seed_len_tol = float(args['--seed-len-tol'])
    seed_angle_tol = float(args['--seed-angle-tol'])
    seed_pair_num = int(args['--seed-pair-num'])
    sort_by = None if sort_by == 'none' else sort_by
    refine = True if refine == 'true' else False
    show_progress = True if show_progress == 'true' else False
    table_file = args['<TABLE-FILE>']
    photon_energy_list = list(
        map(float, args['<PHOTON-ENERGY-LIST>'].split(',')))
    
    table = load_table(table_file)
    table['A0'] = calc_transform_matrix(table['lattice_constants'])

    stop = False
    while not stop:
        job = comm.recv(source=0)
        if job is None:
            comm.send(None, dest=0)  # send dummy response for dummy job
            break
        t0 = time.time()
        res = index(
            table, job['coords'], photon_energy_list, det_dist,
            seed_len_tol=seed_len_tol,
            seed_angle_tol=seed_angle_tol,
            seed_pair_num=seed_pair_num,
            intensity=job['intensity'],
            pre_solution=job['pre_solution'],
            sort=sort_by,
            refine=refine,
            show_progress=show_progress,
        )
        t1 = time.time()
        res['time'] = t1 - t0
        res['id'] = job['id']
        res['nb_peak'] = job['coords'].shape[0]
        comm.send(res, dest=0)
        stop = comm.recv(source=0)
    

if __name__ == "__main__":
    # mpi setup
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    if size == 1:
        print('index script need >= 2 processes!')
        sys.exit()
    rank = comm.Get_rank()
    buffer_size = 1E6

    # parse options
    args = docopt(__doc__)
    if rank == 0:
        master_run(args)
    else:
        worker_run(args)
    
    sys.exit()

        # solution = res['best_solution']
        # A = solution.A_refined
        # print('=' * 100)
        # print('time elapsed %.2f sec' % (t1 - t0))
        # print('searched seed pairs: %d' % res['seed_pair_count'])
        # print('searched solution candidates: %d' % res['solution_count'])   
        # print('match rate(refine refine): %.3f' % solution.match_rate)
        # print('match rate(after refine): %.3f' % solution.match_rate_refined)
        # print('pair_dist(before refine): %.3e' % solution.pair_dist)
        # print('pair_dist(after refine): %.3e' % solution.pair_dist_refined)
        # print('best solution: ', solution.A)
        # print('=' * 100)

        
