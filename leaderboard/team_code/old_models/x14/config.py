import os

class GlobalConfig:
    gpu_id = '0'
    model = 'x14'
    logdir = 'log/'+model+'_w1'
    init_stop_counter = 15

    n_class = 23
    batch_size = 20
    coverage_area = 64 #untuk top view SC, HXW sama dalam meter

    #parameter untuk MGN
    MGN = True
    loss_weights = [1, 1, 1, 1, 1, 1, 1]
    lw_alpha = 1.5
    bottleneck = [335, 718]

	# Data
    seq_len = 1 # jumlah input seq
    pred_len = 3 # future waypoints predicted

    root_dir = '/home/aisl/OSKAR/Transfuser/transfuser_data/clear_noon_full_data'  #14_weathers_full_data clear_noon_full_data
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

    lr = 1e-4 # learning rate #pakai AdamW
    weight_decay = 1e-3

    # Controller
    # control_option = 4 #1-one_of 2-both_must 3-pid_only 4-mlp_only 5-pid_def
    #control weights untuk PID dan MLP dari tuningan MGN
    #urutan steer, throttle, brake
    #baca dulu trainval_log.csv setelah training selesai, dan normalize bobotnya 0-1
    #LWS: lw_wp lw_str lw_thr lw_brk saat convergence
    lws = [1, 1, 1, 1]
    # : [1, 1, 1, 1]
    # _w1 : [1, 1, 1, 1]
    # _t2 : [1, 1, 1, 1]
    # _t2w1 : [1, 1, 1, 1]
    cw_pid = [lws[0]/(lws[0]+lws[1]), lws[0]/(lws[0]+lws[2]), lws[0]/(lws[0]+lws[3])] #str, thrt, brk
    cw_mlp = [1-cw_pid[0], 1-cw_pid[1], 1-cw_pid[2]] #str, thrt, brk


    turn_KP = 1.25
    turn_KI = 0.75
    turn_KD = 0.3
    turn_n = 40 # buffer size

    speed_KP = 5.0
    speed_KI = 0.5
    speed_KD = 1.0
    speed_n = 40 # buffer size

    max_throttle = 0.75 # upper limit on throttle signal value in dataset
    brake_speed = 0.4 # desired speed below which brake is triggered
    brake_ratio = 1.1 # ratio of speed to desired speed at which brake is triggered
    clip_delta = 0.25 # maximum change in speed input to logitudinal controller
    min_act_thrt = 0.2 #minimum nilai suatu throttle dianggap aktif diinjak

    #ORDER DALAM RGB!!!!!!!!
    SEG_CLASSES = {
        'colors'        :[[0, 0, 0], [70, 70, 70], [100, 40, 40], [55, 90, 80], [220, 20, 60],  
                            [153, 153, 153], [157, 234, 50], [128, 64, 128], [244, 35, 232], [107, 142, 35], 
                            [0, 0, 142], [102, 102, 156], [220, 220, 0], [70, 130, 180], [81, 0, 81],
                            [150, 100, 100], [230, 150, 140], [180, 165, 180], [250, 170, 30], [110, 190, 160],
                            [170, 120, 50], [45, 60, 150], [145, 170, 100]], 
        'classes'       : ['None', 'Building', 'Fences', 'Other', 'Pedestrian',
                            'Pole', 'RoadLines', 'Road', 'Sidewalk', 'Vegetation',
                            'Vehicle', 'Wall', 'TrafficSign', 'Sky', 'Ground',
                            'Bridge', 'RailTrack', 'GuardRail', 'TrafficLight', 'Static',
                            'Dynamic', 'Water', 'Terrain']
    }

    n_fmap_b0 = [[32,16], [24], [40], [80,112], [192,320,1280]]
    n_fmap_b1 = [[32,16], [24], [40], [80,112], [192,320,1280]] #sama dengan b0
    n_fmap_b2 = [[32,16], [24], [48], [88,120], [208,352,1408]]
    n_fmap_b3 = [[40,24], [32], [48], [96,136], [232,384,1536]] #lihat underdevelopment/efficientnet.py
    n_fmap_b4 = [[48,24], [32], [56], [112,160], [272,448,1792]]
    #jangan lupa untuk mengganti model torchvision di init model.py

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
