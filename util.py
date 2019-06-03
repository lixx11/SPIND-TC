import math
from math import cos, sin
import numpy as np
from numpy.linalg import norm
import h5py


def calc_angle(v1, v2, len1, len2):
    """
    Calculate angle between v1, v2 vectors.
    :param v1: first vector.
    :param v2: second vector.
    :param len1: length of first vector.
    :param len2: length of second vector.
    :return: angle between v1 and v2 in degrees.
    """
    cos_val = v1.dot(v2) / (len1 * len2)
    cos_val = max(min(cos_val, 1.), -1.)
    angle = math.acos(cos_val) / math.pi * 180.
    return angle


def lattice_constants_to_cartesian_vectors(lattice_constants):
    """
    Calculate a,b,c vectors in cartesian system from lattice constants.
    :param lattice_constants: a,b,c,alpha,beta,gamma lattice constants.
    :return: a, b, c vector
    """
    lattice_constants = np.array(lattice_constants)
    if lattice_constants.shape != (6,):
        raise ValueError('Lattice constants must be 1d array with 6 elements.')
    a, b, c = lattice_constants[:3]
    alpha, beta, gamma = np.deg2rad(lattice_constants[3:])
    av = np.array([a, 0, 0], dtype=float)
    bv = np.array([b * np.cos(gamma), b * np.sin(gamma), 0], dtype=float)
    # calculate vector c
    x = np.cos(beta)
    y = (np.cos(alpha) - x * np.cos(gamma)) / np.sin(gamma)
    z = np.sqrt(1. - x**2. - y**2.)
    cv = np.array([x, y, z], dtype=float)
    cv /= norm(cv)
    cv *= c

    return av, bv, cv


def calc_transform_matrix(lattice_constants):
    """
    Calculate transform matrix from lattice constants.
    :param lattice_constants: a,b,c,alpha,beta,gamma lattice constants in
                              angstrom and degree.
    :param lattice_type: lattice type: P, A, B, C, H
    :return: transform matrix A = [a*, b*, c*]
    """
    av, bv, cv = lattice_constants_to_cartesian_vectors(lattice_constants)
    a_star = (np.cross(bv, cv)) / (np.cross(bv, cv).dot(av))
    b_star = (np.cross(cv, av)) / (np.cross(cv, av).dot(bv))
    c_star = (np.cross(av, bv)) / (np.cross(av, bv).dot(cv))
    A = np.zeros((3, 3), dtype='float')  # transform matrix
    A[:, 0] = a_star
    A[:, 1] = b_star
    A[:, 2] = c_star
    return A


def det2fourier(coords, wavelength, det_dist):
    """
    Convert detector coordinates to fourier coordinates.
    :param coords: detector coordinates, Nx2 array.
    :param wavelength: wavelength in angstrom.
    :param det_dist: detetor distance in meters.
    :return: fourier coordinates, Nx3 array.
    """
    nb_coords = coords.shape[0]
    det_dist = np.ones(nb_coords) * det_dist
    det_dist = det_dist.reshape(-1, 1)
    q1 = np.hstack((coords, det_dist))
    q1_len = norm(q1, axis=1).reshape(-1, 1)
    q1 /= q1_len
    q0 = np.array([0., 0., 1.])
    q = 1. / wavelength * (q1 - q0)
    return q


def calc_wavelength(photon_energy):
    """Convert photon energy to wavelength.
    
    Args:
        photon_energy (float): photon energy in eV.
    """
    h = 4.135667662E-15  # Planck constant in eV*s
    c = 2.99792458E8  # light speed in m/s
    return (h * c) / photon_energy


def load_table(filepath):
    """
    Load hkl reference table.
    :param filepath: path of table file.
    :return: table dict with hkl1, hkl2, len/angle, lattice constants,
             min/max res.
    """
    table_h5 = h5py.File(filepath, 'r')
    table = {
        'hkl1': table_h5['hkl1'][()],
        'hkl2': table_h5['hkl2'][()],
        'len_angle': table_h5['len_angle'][()],
        'lattice_constants': table_h5['lattice_constants'][()],
        'low_res': table_h5['low_res'][()],
        'high_res': table_h5['high_res'][()],
        'space_group': table_h5['space_group'][()],
    }
    return table


def rad2deg(rad):
    return float(rad) / math.pi * 180.


def deg2rad(deg):
    return float(deg) / 180. * math.pi


def axis_angle_to_rotation_matrix(axis, angle):
    """
    Calculate rotation matrix from axis/angle form.
    :param axis: axis vector with 3 elements.
    :param angle: angle in degree.
    :return: rotation matrix R.
    """
    x, y, z = axis / norm(axis)
    angle = deg2rad(angle)
    c, s = math.cos(angle), math.sin(angle)
    R = [[c+x**2.*(1-c), x*y*(1-c)-z*s, x*z*(1-c)+y*s],
         [y*x*(1-c)+z*s, c+y**2.*(1-c), y*z*(1-c)-x*s],
         [z*x*(1-c)-y*s, z*y*(1-c)+x*s, c+z**2.*(1-c)]]
    return np.array(R)


def calc_rotation_matrix(mob_v1, mob_v2, ref_v1, ref_v2):
    """
    Calculate rotation matrix R, thus R.dot(ref_vx) ~ mob_vx.
    :param mob_v1: first mobile vector.
    :param mob_v2: second mobile vector.
    :param ref_v1: first reference vector.
    :param ref_v2: second reference vector.
    :return: rotation matrix R.
    """
    # rotate reference vector plane to  mobile plane
    mob_norm = np.cross(mob_v1, mob_v2)  # norm vector of mobile vectors
    ref_norm = np.cross(ref_v1, ref_v2)  # norm vector of reference vectors
    if min(norm(mob_norm), norm(ref_norm)) == 0.:
        return np.identity(3)  # return dummy matrix if co-linear
    axis = np.cross(ref_norm, mob_norm)
    angle = calc_angle(ref_norm, mob_norm, norm(ref_norm), norm(mob_norm))
    R1 = axis_angle_to_rotation_matrix(axis, angle)
    rot_ref_v1, rot_ref_v2 = R1.dot(ref_v1), R1.dot(ref_v2)
    # rotate reference vectors to mobile vectors approximately
    angle1 = calc_angle(rot_ref_v1, mob_v1, norm(rot_ref_v1), norm(mob_v1))
    angle2 = calc_angle(rot_ref_v2, mob_v2, norm(rot_ref_v2), norm(mob_v2))
    angle = (angle1 + angle2) * 0.5
    axis = np.cross(rot_ref_v1, mob_v1)  # important!!
    R2 = axis_angle_to_rotation_matrix(axis, angle)
    R = R2.dot(R1)
    return R


# def rotation_matrix_to_euler_angles(R):
#     sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
#     singular = sy < 1e-6
#     if not singular:
#         x = rad2deg(math.atan2(R[2, 1], R[2, 2]))
#         y = rad2deg(math.atan2(-R[2, 0], sy))
#         z = rad2deg(math.atan2(R[1, 0], R[0, 0]))
#     else:
#         x = rad2deg(math.atan2(-R[1, 2], R[1, 1]))
#         y = rad2deg(math.atan2(-R[2, 0], sy))
#         z = 0
#     return np.array([x, y, z])


def euler_angles_to_rotation_matrix(euler_angles):
    """Convert XYZ Euler angles to rotation matrix
    """
    t1, t2, t3 = np.deg2rad(euler_angles)
    rm = np.array([
        [cos(t2)*cos(t3), cos(t2)*sin(t3), -sin(t2)],
        [sin(t1)*sin(t2)*cos(t3) - cos(t1)*sin(t3), sin(t1) *
         sin(t2)*sin(t3)+cos(t1)*cos(t3), sin(t1)*cos(t2)],
        [cos(t1)*sin(t2)*cos(t3)+sin(t1)*sin(t3), cos(t1) *
         sin(t2)*sin(t3)-sin(t1)*cos(t3), cos(t1)*cos(t2)]
    ])
    return rm


def rotation_matrix_to_euler_angles(rm):
    t1 = math.atan2(rm[1, 2], rm[2, 2])
    c2 = math.sqrt(rm[0, 0]**2 + rm[0, 1]**2)
    t2 = math.atan2(-rm[0, 2], c2)
    s1, c1 = sin(t1), cos(t1)
    t3 = math.atan2(s1*rm[2, 0]-c1*rm[1, 0], c1*rm[1, 1]-s1*rm[2, 1])
    return rad2deg(t1), rad2deg(t2), rad2deg(t3)
