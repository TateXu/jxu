from mne.io import read_raw_brainvision as rrb

from mne.viz import plot_raw

channel_list = ['P3', 'Pz', 'CP1', 'P2', 'Audio', 'CP3', 'CPz', 'POz', 'CCP2h', 'PPO1h', 'PPO2h', 'CPP4h', 'CPP1h', 'CPP2h', 'C2', 'FCC4h', 'C4', 'CP4', 'CCP6h', 'CP1', 'PO3', 'PO2']
def vhdr_load(filename):
	
	data = rrb(filename, eog=('HEOGL', 'HEOGR', 'VEOGb'), misc='auto', scale=1.0, preload=True, verbose=None)

	return data

	