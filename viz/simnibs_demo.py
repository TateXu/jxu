import simnibs
import pdb 


# simnibs under the root dir

file_root = '/home/jxu/SimNIBS/simnibs_examples/ernie/'

# Initialize structure
opt = simnibs.opt_struct.TDCSoptimize()
# Select the leadfield file
opt.leadfield_hdf = file_root + 'leadfield/ernie_leadfield_EEG10-10_UI_Jurak_2007.hdf5'
# Select a name for the optimization



# Select a maximum total current (in A)
opt.max_total_current = 2e-3
# Select a maximum current at each electrodes (in A)
opt.max_individual_current = 1e-3
# Select a maximum number of active electrodes (optional)
opt.max_active_electrodes = 2


#

single = True
if single:
    opt.name =  file_root + 'optimization/single_target'
    # Define optimization target
    target = opt.add_target()
    # Transfrorm a set of coordinates from MNI space to subject space.
    # The second argument of the mni2subject_coords function
    # is the path to the "m2m_subID" folder.
    #http://sprout022.sprout.yale.edu/mni2tal/mni2tal.html
    target.positions = simnibs.mni2subject_coords([-57, -34, -4], file_root + 'm2m_ernie')  # Left BA21
    target.intensity = 0.2
else:
    opt.name = file_root + 'optimization/multi_target'
    # Target in the left motor cortex
    target_left = opt.add_target()
    target_left.positions = simnibs.mni2subject_coords([-57, -34, 14], file_root + 'm2m_ernie')  # Left BA21    [-56.9804, -34.1086, 38.6099]  # 
    target_left.intensity = 0.2
    # Target in the right motor cortex
    target_right = opt.add_target()
    target_right.positions =  simnibs.mni2subject_coords([57, -34, 14], file_root + 'm2m_ernie') # [56.9804, -24.1086, 38.6099] #
    target_right.intensity = -0.2 # negative value revert the direction



# Run optimization
simnibs.run_simnibs(opt)
