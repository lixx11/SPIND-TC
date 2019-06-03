#!/usr/bin/env python
"""
Generate hkl table.

Usage:
    gen_hkl_table.py <HKL-FILE> [options]

Options:
    -h --help               Show this screen.
    -o FILE                 Specify output table file [default: output.h5].
"""

import sys
import os
from mpi4py import MPI
import numpy as np
from numpy.linalg import norm
import h5py
import time
from tqdm import tqdm
from docopt import docopt
from util import calc_transform_matrix, calc_angle


BATCH_NUM = 200


def parse_header(header):
    for line in header:
        k, v = line[1:-1].split('=')
        k, v = k.strip(), v.strip()
        if k == 'a':
            a = float(v.replace('A', ''))
        elif k == 'b':
            b = float(v.replace('A', ''))
        elif k == 'c':
            c = float(v.replace('A', ''))
        elif k == 'alpha':
            alpha = float(v.replace('deg', ''))
        elif k == 'beta':
            beta = float(v.replace('deg', ''))
        elif k == 'gamma':
            gamma = float(v.replace('deg', ''))
        elif k == 'space group':
            space_group = v 
        elif k == 'high res':
            high_res = float(v.replace('A', ''))
        elif k == 'low res':
            low_res = float(v.replace('A', ''))
    header = {
        'a': a,
        'b': b,
        'c': c,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'space group': space_group,
        'high res': high_res,
        'low res': low_res
    }
    return header


def master_run(jobs, output='table.h5'):
    # distribute jobs
    job_num = len(jobs)
    job_id = 0
    reqs = {}
    workers = set(range(1, size))
    for worker in workers:
        if job_id < job_num:
            job = jobs[job_id]
        else:
            job = []  # dummy job
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
            finished, _ = reqs[worker].test()
            if finished:
                if job_id < job_num:
                    comm.isend(stop, dest=worker)
                    comm.isend(jobs[job_id], dest=worker)
                    print('%d/%d --> %d' % (job_id, job_num, worker), flush=True)
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
            finished, _ = reqs[worker].test()
            if finished:
                stop = True
                comm.isend(stop, dest=worker)
                finished_workers.add(worker)
            else:
                all_processed = False
    
    all_saved = False
    workers = set(range(1, size))
    finished_workers = set()
    for worker in workers:
        reqs[worker] = comm.irecv(source=worker)
    while not all_saved:
        time.sleep(0.001)
        all_saved = True
        workers -= finished_workers
        for worker in workers:
            finished, _ = reqs[worker].test()
            if finished:
                stop = True
                comm.isend(stop, dest=worker)
                finished_workers.add(worker)
            else:
                all_saved = False
    
    # merge tables
    table = h5py.File(output, 'w')
    workers = set(range(1, size))
    hkl1_list = []
    hkl2_list = []
    len_angle_list = []
    len_list = []
    for worker in workers:
        worker_file = 'table-tmp-%d.h5' % worker
        sub_table = h5py.File(worker_file, 'r')
        hkl1_list.append(sub_table['hkl1'][()])
        hkl2_list.append(sub_table['hkl2'][()])
        len_angle_list.append(sub_table['len_angle'][:, :3])
        len_list.append(sub_table['len_angle'][:, 3])
        os.remove(worker_file)

    hkl1 = np.concatenate(hkl1_list)
    hkl2 = np.concatenate(hkl2_list)
    len_angle = np.concatenate(len_angle_list)
    len_mean = np.concatenate(len_list)
    sorted_ids = np.argsort(len_mean)

    table.create_dataset('hkl1', data=hkl1[sorted_ids].astype(np.int8))
    table.create_dataset('hkl2', data=hkl2[sorted_ids].astype(np.int8))
    table.create_dataset(
        'len_angle', data=len_angle[sorted_ids].astype(np.float32))
    table.create_dataset('lattice_constants', data=lattice_constants)
    table.create_dataset('low_res', data=header['low res'])
    table.create_dataset('high_res', data=header['high res'])
    table.create_dataset('space_group', data=header['space group'])


def worker_run():
    stop = False
    count = 0
    chunk = []
    chunk_size = 10000

    table =  h5py.File('table-tmp-%d.h5' % rank)
    while not stop:
        job = comm.recv(source=0)
        for i in job:
            for j in range(i+1, len(hkl_array)):
                q1, q2 = q_vectors[i], q_vectors[j]
                len1, len2 = q_lengths[i], q_lengths[j]
                len_mean = (len1 + len2) * 0.5
                hkl1, hkl2 = hkl_array[i], hkl_array[j]
                angle = calc_angle(q1, q2, len1, len2)
                if len1 >= len2:
                    row = [hkl1[0], hkl1[1], hkl1[2],
                           hkl2[0], hkl2[1], hkl2[2],
                           len1, len2, angle, len_mean]
                else:
                    row = [hkl2[0], hkl2[1], hkl2[2],
                           hkl1[0], hkl1[1], hkl1[2],
                           len2, len1, angle, len_mean]
                chunk.append(row)
                count += 1
                if count % chunk_size == 0:
                    save_chunk(chunk, table)
                    chunk = []
        comm.send(job, dest=0)
        stop = comm.recv(source=0)
    # last chunk
    if len(chunk) > 0:
        save_chunk(chunk, table)
    table.close()

    done = True
    comm.send(done, dest=0)


def save_chunk(chunk, table):
    chunk_size = len(chunk)
    chunk = np.array(chunk)
    if 'hkl1' in table.keys():  # dataset existed
        n = table['hkl1'].shape[0]
        table['hkl1'].resize(n+chunk_size, axis=0)
        table['hkl1'][n:n+chunk_size] = chunk[:, 0:3].astype(np.int8)
        table['hkl2'].resize(n+chunk_size, axis=0)
        table['hkl2'][n:n+chunk_size] = chunk[:, 3:6].astype(np.int8)
        table['len_angle'].resize(n+chunk_size, axis=0)
        table['len_angle'][n:n+chunk_size] = chunk[:, 6:10].astype(np.float32)
    else:  # dataset not existed, create it
        table.create_dataset(
            'hkl1', data=chunk[:, 0:3].astype(np.int8), maxshape=(None, 3))
        table.create_dataset(
            'hkl2', data=chunk[:, 3:6].astype(np.int8), maxshape=(None, 3))
        table.create_dataset(
            'len_angle', data=chunk[:, 6:10].astype(np.float32),
            maxshape=[None, 4])


if __name__ == '__main__':
    # mpi setup
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    if size == 1:
        print('gen_hkl_table script need >= 2 processes!')
        sys.exit()
    rank = comm.Get_rank()

    args = docopt(__doc__)
    hkl_file = args['<HKL-FILE>']
    table_file = args['-o']
    # load hkl file
    with open(hkl_file) as f:
        contents = f.readlines()
    # parse header
    header = [line for line in contents if line.startswith('#')]
    print('HKL file header:\n%s' % ''.join(header))
    header = parse_header(header)
    lattice_constants = [
        header['a'], header['b'], header['c'], 
        header['alpha'], header['beta'], header['gamma']
    ]
    A = calc_transform_matrix(lattice_constants)
    # generate and check hkl array
    hkl_array = np.array([list(map(int, line.split())) for line in contents if not line.startswith('#')])
    hkl_array = np.concatenate([hkl_array, -hkl_array])
    if hkl_array.min() < -128 or hkl_array.max() > 127:
        print('hkl out of int8, please consider reducing resolution range.')
        sys.exit()
    
    q_vectors = A.dot(hkl_array.T).T
    q_lengths = norm(q_vectors, axis=1)

    if rank == 0:
        print('hkl orders: %d' % hkl_array.shape[0])
        # group jobs
        raw_jobs = np.arange(hkl_array.shape[0])  # raw jobs
        grouped_jobs = []  # grouped jobs
        for i in range(len(raw_jobs) // 2):
            grouped_jobs.append([raw_jobs[i], raw_jobs[-i-1]])
        if len(raw_jobs) % 2 == 1:
            mid_id = len(raw_jobs) // 2
            grouped_jobs.append([raw_jobs[mid_id]])
        # split grouped jobs into big batches
        job_batches = []
        batches = np.array_split(np.arange(len(grouped_jobs)), 200)
        for batch in batches:
            if len(batch) > 0:
                job_batch = []
                for i in range(len(batch)):
                    job_batch += grouped_jobs[batch[i]]
                job_batches.append(job_batch)
        master_run(job_batches, output=table_file)
    else:
        worker_run()