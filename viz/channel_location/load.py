from scipy.io import loadmat
import numpy as np
from scipy.spatial import distance_matrix

import site


class electrodes():
    def __init__(self, root='./', select_field=['labels', 'X', 'Y', 'Z']):
        self.root = root
        self.select_field = select_field

    def MunichMI(self):
        self.root = site.getsitepackages()[0]
        struct_file = loadmat(self.root + '/jxu/viz/channel_location/Munich128.mat')
        # channel_fields = [*struct_file['Chanlocs'][0][0].dtype.fields.keys()]
        nchan = len(struct_file['Chanlocs'][0])
        loc = np.empty((nchan, len(self.select_field)))

        for id_field, field in enumerate(self.select_field):
            if field =='labels':
                loc[:, id_field] = np.int16(np.squeeze(np.concatenate(struct_file['Chanlocs'][field][0])))
            else:
                loc[:, id_field] = np.squeeze(np.concatenate(struct_file['Chanlocs'][field][0]))

        return self.select_field, loc


        

