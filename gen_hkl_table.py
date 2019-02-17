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
import numpy as np
from numpy.linalg import norm
import h5py
from tqdm import tqdm
from docopt import docopt
from util import calc_transform_matrix, calc_angle

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


if __name__ == '__main__':
    args = docopt(__doc__)
    hkl_file = args['<HKL-FILE>']
    table_file = args['-o']
    with open(hkl_file) as f:
        contents = f.readlines()
    header = [line for line in contents if line.startswith('#')]
    print('HKL file header:\n%s' % ''.join(header))
    header = parse_header(header)

    lattice_constants = [
        header['a'], header['b'], header['c'], 
        header['alpha'], header['beta'], header['gamma']
    ]
    A = calc_transform_matrix(lattice_constants)
    hkl = np.array([list(map(int, line.split())) for line in contents if not line.startswith('#')])
    hkl = np.concatenate([hkl, -hkl])
    
    q_vectors = A.dot(hkl.T).T
    q_lengths = norm(q_vectors, axis=1)

    hkl_data = []
    for i in tqdm(range(hkl.shape[0])):
        for j in range(i+1, hkl.shape[0]):
            q1, q2 = q_vectors[i], q_vectors[j]
            len1, len2 = q_lengths[i], q_lengths[j]
            len_avg = (len1 + len2) * 0.5
            hkl1, hkl2 = hkl[i], hkl[j]
            angle = calc_angle(q1, q2, len1, len2)
            if len1 >= len2:
                row = [
                    hkl1[0], hkl1[1], hkl1[2],
                    hkl2[0], hkl2[1], hkl2[2],
                    len1, len2, angle, len_avg
                ]
            else:
                row = [
                    hkl2[0], hkl2[1], hkl2[2],
                    hkl1[0], hkl1[1], hkl1[2],
                    len2, len1, angle, len_avg
                ]
            hkl_data.append(row)
    hkl_data.sort(key=lambda x: x[9])
    hkl_data = np.array(hkl_data)
    # save hkl table to h5 file
    table_h5 = h5py.File(table_file)
    table_h5.create_dataset('hkl1', data=hkl_data[:,0:3].astype('int'))
    table_h5.create_dataset('hkl2', data=hkl_data[:,3:6].astype('int'))
    table_h5.create_dataset(
        'len_angle', data=hkl_data[:,6:9].astype('float'))
    table_h5.create_dataset('lattice_constants', data=lattice_constants)
    table_h5.create_dataset('low_res', data=header['low res'])
    table_h5.create_dataset('high_res', data=header['high res'])
    table_h5.create_dataset('space_group', data=header['space group'])
    table_h5.close()
    print('HKL table saved to %s' % table_file)
