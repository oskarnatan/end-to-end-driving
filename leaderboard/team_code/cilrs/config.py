import os

class GlobalConfig:
	# Data
    seq_len = 1 # input timesteps
    pred_len = 0 # future waypoints predicted, not required for CILRS

    root_dir = '/home/oskar/OSKAR/Transfuser/transfuser_data/14_weathers_full_data'
    train_towns = ['Town01', 'Town02', 'Town03', 'Town04', 'Town06', 'Town07', 'Town10']
    val_towns = ['Town05']
    # train_towns = ['Town00']
    # val_towns = ['Town00']
    train_data, val_data = [], []
    for town in train_towns:
        if not (town == 'Town07' or town == 'Town10'):
            train_data.append(os.path.join(root_dir, town+'_long'))
        train_data.append(os.path.join(root_dir, town+'_short'))
        train_data.append(os.path.join(root_dir, town+'_tiny'))
        # train_data.append(os.path.join(root_dir, town+'_x'))
    for town in val_towns:
        # val_data.append(os.path.join(root_dir, town+'_long'))
        val_data.append(os.path.join(root_dir, town+'_short'))
        val_data.append(os.path.join(root_dir, town+'_tiny'))
        # val_data.append(os.path.join(root_dir, town+'_x'))

    ignore_sides = True # don't consider side cameras
    ignore_rear = True # don't consider rear cameras

    input_resolution = 256

    scale = 1 # image pre-processing
    crop = 256 # image pre-processing

    max_throttle = 0.75 # upper limit on throttle signal value in dataset

    lr = 1e-4 # learning rate

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
