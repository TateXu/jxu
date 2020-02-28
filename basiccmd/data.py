import numpy as np


def whosmy(*args):
  sequentialTypes = [dict, list, tuple] 
  for var in args:
    t=type(var)
    if t== np.ndarray:  
      print type(var),var.dtype, var.shape
    elif t in sequentialTypes: 
      print type(var), len(var)
    else:
      print type(var)
def save_loop(self, save_folder, save_file, save_data):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    cnt = 0
    while os.path.exists(save_folder + save_file + str(cnt) + '.yml'):
        cnt += 1
    with open(save_folder + save_file + str(cnt) + '.yml', 'w') as infile:
        yaml.dump(save_data, infile, default_flow_style=False)
