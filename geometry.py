#！/usr/bin/env python

import numpy as np 
from numpy.linalg import inv
import h5py
import matplotlib.pyplot as plt


class Panel(object):
    def __init__(self, name):
        self.name = name
        self.prop = {}

    def __str__(self):
        return self.name


class Detector(object):
    def __init__(self, geom_file, panel_names):
        self.geom_file = geom_file
        self.panel_names = panel_names
        self.xmap, self.ymap = geom2map(geom_file, panel_names)
        self.xoffset = self.xmap.min()
        self.yoffset = self.ymap.min()

    def assemble(self, img_raw):
        """
        convert raw image to assembled image.
        """
        xmap = (self.xmap - self.xmap.min()).round().astype(int)
        ymap = (self.ymap - self.ymap.min()).round().astype(int)
        img_size = (ymap.max() - ymap.min() + 1, xmap.max() - xmap.min() + 1)
        img_assembled = np.zeros(img_size)
        img_assembled[ymap, xmap] = img_raw
        return img_assembled

    def map2xy(self, coord_fs, coord_ss, shift_center=False):
        """
        convert fs/ss coordinates to xy coordinates.
        """
        coord_fs = np.array(coord_fs)
        coord_ss = np.array(coord_ss)
        coord_fs = coord_fs.round().astype(int)
        coord_ss = coord_ss.round().astype(int)
        coord_x = self.xmap[coord_ss, coord_fs]
        coord_y = self.ymap[coord_ss, coord_fs]
        if shift_center:
            coord_x -= self.xoffset
            coord_y -= self.yoffset
        return coord_x, coord_y


def geom2map(geom_file, panel_names):
    with open(geom_file, 'r') as f:
        contents = f.readlines()
    panels = [Panel(name) for name in panel_names]
    # parse geom file
    for content in contents:
        for panel in panels:
            if content.startswith(panel.name):
                prop, value = content.split('=')
                prop = prop.split('/')[1].strip()
                if prop not in ('fs', 'ss'):
                    value = float(value.strip())
                else:
                    value = value.strip()
                panel.prop[prop] = value
    # construct transform map
    max_fs = int(max([p.prop['max_fs'] for p in panels]))
    max_ss = int(max([p.prop['max_ss'] for p in panels]))
    xmap = np.zeros((max_ss+1, max_fs+1))
    ymap = xmap.copy()
    for panel in panels:
        min_fs = int(panel.prop['min_fs'])
        min_ss = int(panel.prop['min_ss'])
        max_fs = int(panel.prop['max_fs'])
        max_ss = int(panel.prop['max_ss'])
        grid_fs, grid_ss = np.meshgrid(
            range(min_fs, max_fs + 1), range(min_ss, max_ss + 1))
        grid_fs, grid_ss = grid_fs.flatten(), grid_ss.flatten()
        grid = np.vstack([grid_ss, grid_fs])
        xy2fs_coef = list(map(float, panel.prop['fs'].strip()[:-1].split('x')))
        xy2ss_coef = list(map(float, panel.prop['ss'].strip()[:-1].split('x')))
        M = inv(np.vstack([xy2ss_coef, xy2fs_coef]))

        map_xy = M.dot(grid)
        map_x = map_xy[0, :]
        map_y = map_xy[1, :]
        xmap[grid_ss, grid_fs] = map_x - map_x[0] + panel.prop['corner_x']
        ymap[grid_ss, grid_fs] = map_y - map_y[0] + panel.prop['corner_y']

    return xmap, ymap


if __name__ == "__main__":
    geom_file = 'data/geom/7keV.geom'
    data_file = 'data/exp1/tag-186380316.h5'
    detector = detector = Detector(geom_file, ['q%d' % i for i in range(1, 9)])
    data = h5py.File(data_file, 'r')
    image = data['data'][()]
    peaks_raw = data['residual_points_9keV'][()]
    peaks_x, peaks_y = detector.map2xy(peaks_raw[:, 0], peaks_raw[:, 1], shift_center=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(peaks_x, peaks_y, marker='o',
               edgecolors='red', facecolors='none')
    im = ax.imshow(detector.assemble(image))
    im.set_clim(0, 400)
    plt.show()
