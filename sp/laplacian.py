from scipy.spatial import distance_matrix
from jxu.viz.channel_location.load import electrodes
import numpy as np


def surf_LP(data, loc, filter_type='small'):
    # Spatial filter selection for EEG-based communication, McFarland& Wolpaw, (1997)
    label, loc = electrodes().MunichMI()
    dist_mat = distance_matrix(loc[:, 1:], loc[:, 1:], p=2)

    ranked_dist = np.argsort(dist_mat, axis=1)

    if filter_type == 'small':
        # Small LP
        adjacent_elec = ranked_dist[:, :5]
    elif filter_type == 'large':
        # Large LP
        adjacent_elec = np.concatenate((ranked_dist[:, 0:1], ranked_dist[:, 5:9]), axis=1)

    filtered_data = np.empty(data.shape)
    for row in range(data.shape[0]):
        filtered_data[row] = data[adjacent_elec[row, 0]] - data[adjacent_elec[row, 1:]].sum(axis=0) / 4

    return filtered_data


