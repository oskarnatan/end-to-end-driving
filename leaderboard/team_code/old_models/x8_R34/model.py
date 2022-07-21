from collections import deque
import sys
import numpy as np
from torch import torch, cat, add, nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms




#FUNGSI INISIALISASI WEIGHTS MODEL
#baca https://pytorch.org/docs/stable/nn.init.html
#kaiming he
def kaiming_w_init(layer, nonlinearity='relu'):
    nn.init.kaiming_normal_(layer.weight, nonlinearity=nonlinearity)
    layer.bias.data.fill_(0.01)

class ConvBNRelu(nn.Module):
    def __init__(self, channelx, stridex=1, kernelx=3, paddingx=1):
        super(ConvBNRelu, self).__init__()
        self.conv = nn.Conv2d(channelx[0], channelx[1], kernel_size=kernelx, stride=stridex, padding=paddingx, padding_mode='zeros')
        self.bn = nn.BatchNorm2d(channelx[1])
        self.relu = nn.ReLU()
        #weights initialization
        kaiming_w_init(self.conv)
    
    def forward(self, x):
        x = self.conv(x) 
        x = self.bn(x) 
        y = self.relu(x)
        return y

class ConvBlock(nn.Module):
    def __init__(self, channel, final=False): #up, 
        super(ConvBlock, self).__init__()
        #conv block
        if final:
            self.conv_block0 = ConvBNRelu(channelx=[channel[0], channel[0]], stridex=1)
            self.conv_block1 = nn.Sequential(
            nn.Conv2d(channel[0], channel[1], kernel_size=1),
            nn.Sigmoid()
            )
        else:
            self.conv_block0 = ConvBNRelu(channelx=[channel[0], channel[1]], stridex=1)
            self.conv_block1 = ConvBNRelu(channelx=[channel[1], channel[1]], stridex=1)
 
    def forward(self, x):
        #convolutional block
        y = self.conv_block0(x)
        y = self.conv_block1(y)
        return y


class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D
        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0
    
    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)
        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0
        out_control = self._K_P * error + self._K_I * integral + self._K_D * derivative
        return out_control



class x8_R34(nn.Module): #
    #default input channel adalah 3 untuk RGB, 2 untuk DVS, 1 untuk LiDAR
    def __init__(self, config, device):#n_fmap, n_class=[23,10], n_wp=5, in_channel_dim=[3,2], spatial_dim=[240, 320], gpu_device=None): 
        super(x8_R34, self).__init__()
        self.config = config
        n_fmap_r34 = [[64], [64], [128], [256], [512]] #lihat underdevelopment/resnet.py
        self.gpu_device = device
        self.flatten = nn.Flatten()
        #------------------------------------------------------------------------------------------------
        #RGB, jika inputnya sequence, maka jumlah input channel juga harus menyesuaikan
        self.rgb_normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.RGB_encoder = models.resnet34(pretrained=True) #resnet34 resnet18
        self.RGB_encoder.fc = nn.Sequential() #cara paling gampang untuk menghilangkan fc layer yang tidak diperlukan
        self.RGB_encoder.avgpool = nn.Sequential()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  
        #SS
        self.conv3_ss_f = ConvBlock(channel=[n_fmap_r34[4][-1]+n_fmap_r34[3][-1], n_fmap_r34[3][-1]])#, up=True)
        self.conv2_ss_f = ConvBlock(channel=[n_fmap_r34[3][-1]+n_fmap_r34[2][-1], n_fmap_r34[2][-1]])#, up=True)
        self.conv1_ss_f = ConvBlock(channel=[n_fmap_r34[2][-1]+n_fmap_r34[1][-1], n_fmap_r34[1][-1]])#, up=True)
        self.conv0_ss_f = ConvBlock(channel=[n_fmap_r34[1][-1]+n_fmap_r34[0][-1], n_fmap_r34[0][0]])#, up=True)
        self.final_ss_f = ConvBlock(channel=[n_fmap_r34[0][0], config.n_class], final=True)#, up=False)
        #------------------------------------------------------------------------------------------------
        #untuk semantic cloud generator
        self.cover_area = config.coverage_area
        self.n_class = config.n_class
        self.h, self.w = config.input_resolution, config.input_resolution
        fx = 160 #from cam proj matrix (cam info)
        self.x_matrix = torch.vstack([torch.arange(-self.w/2, self.w/2)]*self.h) / fx
        self.x_matrix = self.x_matrix.to(device)
        #SC
        self.SC_encoder = models.resnet34(pretrained=False) #resnet34 resnet18
        self.SC_encoder.conv1 = nn.Conv2d(config.n_class, n_fmap_r34[0][0], kernel_size=7, stride=2, padding=3, bias=False) #ganti input channel conv pertamanya, buat SC cloud
        self.SC_encoder.fc = nn.Sequential() #cara paling gampang untuk menghilangkan fc layer yang tidak diperlukan
        self.SC_encoder.avgpool = nn.Sequential()
        #------------------------------------------------------------------------------------------------
        #waypoint predition, input size selalu 4 karena concat dari x,y wp dan x,y rooute next
        self.n_wp = config.pred_len
        #jumlahkan kedua bottleneck
        self.necks_net = nn.Sequential( #inputnya sum dari 2 bottleneck
            ConvBNRelu(channelx=[n_fmap_r34[4][-1], n_fmap_r34[4][-1]], kernelx=1, paddingx=0, stridex=1),
            # STNet(ch=[n_fmap_r34[4][1], n_fmap_r34[3][-1], n_fmap_r34[2][-1]]),
            ConvBNRelu(channelx=[n_fmap_r34[4][-1], n_fmap_r34[4][-1]], kernelx=2, paddingx=0, stridex=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(n_fmap_r34[4][-1], n_fmap_r34[3][0]),
            nn.BatchNorm1d(n_fmap_r34[3][0]),
            nn.ReLU()
        )
        #embedding hidden state dari vehicle location, next route, and velocity
        self.gru_nxr = nn.GRUCell(input_size=2, hidden_size=n_fmap_r34[3][0]) #gru untuk next route
        self.gru_vel = nn.GRUCell(input_size=1, hidden_size=n_fmap_r34[3][0]) #gru untuk current velocity

        #wp predictor
        self.gru_wp = nn.GRUCell(input_size=2, hidden_size=n_fmap_r34[3][0])#nn.ModuleList([nn.GRUCell(input_size=2, hidden_size=n_fmap_r34[2][-1]) for _ in range(config.pred_len)])
        self.pred_dwp = nn.Linear(n_fmap_r34[3][0], 2)

        #controller
        #MLP Controller
        self.controller = nn.Sequential(
            nn.Linear(n_fmap_r34[3][0], n_fmap_r34[2][-1]),
            nn.ReLU(),
            nn.Linear(n_fmap_r34[2][-1], n_fmap_r34[1][0]),
            nn.ReLU(),
            nn.Linear(n_fmap_r34[1][0], 3),
            nn.ReLU()
        )
        #PID Controller
        self.turn_controller = PIDController(K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n)
        self.speed_controller = PIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n)
        

    def forward(self, rgb_f, depth_f, next_route, velo_in):#, gt_ss):
        #------------------------------------------------------------------------------------------------
        #bagian downsampling
        RGB_features_sum = 0
        for i in range(self.config.seq_len): #loop semua input dalam buffer
            in_rgb = self.rgb_normalizer(rgb_f[i])
            x_RGB = self.RGB_encoder.conv1(in_rgb)
            x_RGB = self.RGB_encoder.bn1(x_RGB)
            RGB_features0 = self.RGB_encoder.relu(x_RGB)
            RGB_features1 = self.RGB_encoder.layer1(self.RGB_encoder.maxpool(RGB_features0))
            RGB_features2 = self.RGB_encoder.layer2(RGB_features1)
            RGB_features3 = self.RGB_encoder.layer3(RGB_features2)
            RGB_features4 = self.RGB_encoder.layer4(RGB_features3)
            RGB_features_sum += RGB_features4
        #bagian upsampling
        ss_f_3 = self.conv3_ss_f(cat([self.up(RGB_features4), RGB_features3], dim=1))
        ss_f_2 = self.conv2_ss_f(cat([self.up(ss_f_3), RGB_features2], dim=1))
        ss_f_1 = self.conv1_ss_f(cat([self.up(ss_f_2), RGB_features1], dim=1))
        ss_f_0 = self.conv0_ss_f(cat([self.up(ss_f_1), RGB_features0], dim=1))
        ss_f = self.final_ss_f(self.up(ss_f_0))
        #------------------------------------------------------------------------------------------------
        #buat semantic cloud
        top_view_sc = self.gen_top_view_sc(depth_f[-1], ss_f) #ingat, depth juga sequence, ambil yang terakhir
        #bagian downsampling
        x_SC = self.SC_encoder.conv1(top_view_sc)
        x_SC = self.SC_encoder.bn1(x_SC)
        SC_features0 = self.SC_encoder.relu(x_SC)
        SC_features1 = self.SC_encoder.layer1(self.SC_encoder.maxpool(SC_features0))
        SC_features2 = self.SC_encoder.layer2(SC_features1)
        SC_features3 = self.SC_encoder.layer3(SC_features2)
        SC_features4 = self.SC_encoder.layer4(SC_features3)
        #------------------------------------------------------------------------------------------------
        #waypoint prediction
        #get hidden state dari gabungan kedua bottleneck
        hid_states = self.necks_net(RGB_features_sum+SC_features4) #hid_state0)# cat([RGB_features_sum, SC_features4], dim=1)
        #hid_states dari next route dan velocity
        hid_state_nxr = self.gru_nxr(next_route, hid_states)
        hid_state_vel = self.gru_vel(torch.reshape(velo_in, (velo_in.shape[0], 1)), hid_states)

        #jumlahkan hidden state
        hx = hid_states+hid_state_nxr+hid_state_vel
        # initial input car location ke GRU, selalu buat batch size x 2 (0,0) (xy)
        xy = torch.zeros(size=(hx.shape[0], 2)).float().to(self.gpu_device)
        #predict delta wp
        out_wp = list()
        for _ in range(self.n_wp):
            hx = self.gru_wp(xy, hx)
            d_xy = self.pred_dwp(hx) 
            xy = xy + d_xy
            out_wp.append(xy)
        pred_wp = torch.stack(out_wp, dim=1)

        #control decoder
        control_pred = self.controller(hx) #cat([hid_states, hid_state_nxr, hid_state_vel], dim=1)
        steer = control_pred[:,0] * 2 - 1. # convert from [0,1] to [-1,1]
        throttle = control_pred[:,1] * self.config.max_throttle
        brake = control_pred[:,2]

        return ss_f, pred_wp, steer, throttle, brake, top_view_sc


    def gen_top_view_sc(self, depth, semseg):
        #proses awal
        depth_in = depth * 1000.0 #normalisasi ke 1 - 1000
        _, label_img = torch.max(semseg, dim=1) #pada axis C
        cloud_data_n = torch.ravel(torch.tensor([[n for _ in range(self.h*self.w)] for n in range(depth.shape[0])])).to(self.gpu_device)
        # cloud_data_x = torch.ravel(depth_in * self.x_matrix)
        # cloud_data_z = torch.ravel(depth_in)
        # cloud_data_cls = torch.ravel(label_img)
        
        #normalize ke frame 
        cloud_data_x = torch.round(((depth_in * self.x_matrix) + (self.cover_area/2)) * (self.w-1) / self.cover_area).ravel()
        cloud_data_z = torch.round((depth_in * -(self.h-1) / self.cover_area) + (self.h-1)).ravel()

        #cari index interest
        bool_xz = torch.logical_and(torch.logical_and(cloud_data_x <= self.w-1, cloud_data_x >= 0), torch.logical_and(cloud_data_z <= self.h-1, cloud_data_z >= 0))
        idx_xz = bool_xz.nonzero().squeeze() #hilangkan axis dengan size=1, sehingga tidak perlu nambahkan ".item()" nantinya

        #stack n x z cls dan plot
        coorx = torch.stack([cloud_data_n, label_img.ravel(), cloud_data_z, cloud_data_x])
        coor_clsn = torch.unique(coorx[:, idx_xz], dim=1).long() #tensor harus long supaya bisa digunakan sebagai index
        # coor_clsn = torch.stack([self.cloud_data_n[idx_xz], cloud_data_cls[idx_xz], cloud_data_z[idx_xz], cloud_data_x[idx_xz]])
        # coor_clsn = torch.unique(coor_clsn, dim=1).type(torch.long) #tensor harus long supaya bisa digunakan sebagai index
        # top_view_sc = torch.zeros((depth.shape[0], self.n_class, self.h, self.w)).float().to(self.gpu_device)   
        top_view_sc = torch.zeros_like(semseg) #ini lebih cepat karena secara otomatis size, tipe data, dan device sama dengan yang dimiliki inputnya (semseg)
        top_view_sc[coor_clsn[0], coor_clsn[1], coor_clsn[2], coor_clsn[3]] = 1.0 #format axis dari NCHW
        # for j in range(coor_clsn.shape[1]):
        #     top_view_sc[coor_clsn[0][j]][coor_clsn[1][j]][coor_clsn[2][j]][coor_clsn[3][j]] = 1.0 #tidak perlu ".item()"

        return top_view_sc


    def mlp_pid_control(self, waypoints, velocity, mlp_steer, mlp_throttle, mlp_brake, cw=[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]):
        assert(waypoints.size(0)==1)
        waypoints = waypoints[0].data.cpu().numpy()

        # flip y is (forward is negative in our waypoints)
        waypoints[:,1] *= -1
        speed = velocity[0].data.cpu().numpy()

        aim = (waypoints[1] + waypoints[0]) / 2.0
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        pid_steer = self.turn_controller.step(angle)
        pid_steer = np.clip(pid_steer, -1.0, 1.0)

        desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0
        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        pid_throttle = self.speed_controller.step(delta)
        pid_throttle = np.clip(pid_throttle, 0.0, self.config.max_throttle)
        pid_brake = 0.0
        #constrain dari brake flag
        # if desired_speed < self.config.brake_speed or (speed/desired_speed) > self.config.brake_ratio:
        #     pid_throttle = 0.0
        #     pid_brake = 1.0
        # else: #jika tidak maka ya jalan seperti biasanya
        #     pid_brake = 0.0

        #final decision
        if self.config.control_option == 1:
            #opsi 1: jika salah satu controller aktif (>0.1), maka vehicle jalan. vehicle berhenti jika kedua controller non aktif (<0.1)
            steer = np.clip(cw[0][0]*pid_steer + cw[1][0]*mlp_steer, -1.0, 1.0)
            throttle = np.clip(cw[0][1]*pid_throttle + cw[1][1]*mlp_throttle, 0.0, self.config.max_throttle)
            brake = 0.0
            if (pid_throttle >= self.config.min_act_thrt) and (mlp_throttle < self.config.min_act_thrt):
                steer = pid_steer
                throttle = pid_throttle
            elif (pid_throttle < self.config.min_act_thrt) and (mlp_throttle >= self.config.min_act_thrt):
                steer = mlp_steer
                throttle = mlp_throttle
            elif (pid_throttle < self.config.min_act_thrt) and (mlp_throttle < self.config.min_act_thrt):
                throttle = 0.0
                pid_brake = 1.0
                brake = np.clip(cw[0][2]*pid_brake + cw[1][2]*mlp_brake, 0.0, 1.0)
        elif self.config.control_option == 2:
            #opsi 2: vehicle jalan jika dan hanya jika kedua controller aktif. jika salah satu saja non aktif, maka vehicle berhenti
            steer = np.clip(cw[0][0]*pid_steer + cw[1][0]*mlp_steer, -1.0, 1.0)
            throttle = np.clip(cw[0][1]*pid_throttle + cw[1][1]*mlp_throttle, 0.0, self.config.max_throttle)
            brake = 0.0
            if (pid_throttle < self.config.min_act_thrt) or (mlp_throttle < self.config.min_act_thrt):
                throttle = 0.0
                pid_brake = 1.0
                brake = np.clip(cw[0][2]*pid_brake + cw[1][2]*mlp_brake, 0.0, 1.0)
        elif self.config.control_option == 3:
            #opsi 3: PID only
            steer = pid_steer
            throttle = pid_throttle
            brake = 0.0
            if pid_throttle < self.config.min_act_thrt:
                throttle = 0.0
                pid_brake = 1.0
                brake = pid_brake
        elif self.config.control_option == 4:
            #opsi 4: MLP only
            steer = mlp_steer
            throttle = mlp_throttle
            brake = 0.0
            if mlp_throttle < self.config.min_act_thrt:
                throttle = 0.0
                pid_brake = 1.0
                brake = mlp_brake
        elif self.config.control_option == 5:
            #opsi 5: PID def
            steer = pid_steer
            throttle = pid_throttle
            brake = 0.0
            if desired_speed < self.config.brake_speed or (speed/desired_speed) > self.config.brake_ratio:
                throttle = 0.0
                pid_brake = 1.0
                brake = pid_brake
        else:
            sys.exit("ERROR, FALSE CONTROL OPTION")

        metadata = {
            'speed': float(speed.astype(np.float64)),
            'steer': float(steer),
            'throttle': float(throttle),
            'brake': float(brake),
            'cw_pid': [float(cw[0][0]), float(cw[0][1]), float(cw[0][2])],
            'pid_steer': float(pid_steer),
            'pid_throttle': float(pid_throttle),
            'pid_brake': float(pid_brake),
            'cw_mlp': [float(cw[1][0]), float(cw[1][1]), float(cw[1][2])],
            'mlp_steer': float(mlp_steer),
            'mlp_throttle': float(mlp_throttle),
            'mlp_brake': float(mlp_brake),
            # 'wp_4': tuple(waypoints[3].astype(np.float64)), #tambahan
            'wp_3': tuple(waypoints[2].astype(np.float64)), #tambahan
            'wp_2': tuple(waypoints[1].astype(np.float64)),
            'wp_1': tuple(waypoints[0].astype(np.float64)),
            'desired_speed': float(desired_speed.astype(np.float64)),
            'angle': float(angle.astype(np.float64)),
            'aim': tuple(aim.astype(np.float64)),
            'delta': float(delta.astype(np.float64)),
            'car_pos': None, #akan direplace di fungsi agent
            'next_point': None, #akan direplace di fungsi agent
        }
        return steer, throttle, brake, metadata



"""

    def gen_top_view_sc(self, depth, semseg):
        # print(depth[0][0].shape)
        # print(depth.shape[0])
        # print(semseg.shape)
        #camera proj matrix --> [160.00000000000003, 0.0, 160.0, 0.0, 0.0, 160.00000000000003, 120.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        sc_batch = []
        #generate common cloud
        #iterasi pada setiap batch data
        for i in range(depth.shape[0]):
            cloud_data = np.ones(self.h * self.w, dtype=self.field_types)
            #pakai ini jika yang diprediksi adalah depth dalam 3 channel
            # de_f_rgb = depth[i] * 255.0 #normalisasi ke 0 - 255 dulu    .cpu().data.numpy()
            # de_f_rgb = de_f_rgb.permute(1,2,0) #pindah ke channel last
            # # arrayd = de_f_rgb.float32() #astype(np.float32)
            # # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
            # normalized_depth = torch.matmul(de_f_rgb, self.depth_normalizer) #np.dot
            # depth_in = normalized_depth*1000.0/16777215.0  # (256.0 * 256.0 * 256.0 - 1.0) 
            # # depth_in = torch.from_numpy(depthx*1000).to(device) #--> rangenya 0 - 1000
            #pakai ini jika yang diprediksi adalah depth dalam 1 channel
            depth_in = depth[i][0] * 1000.0 #normalisasi ke 1 - 1000
            #proses depth
            X = depth_in * self.x_matrix
            cloud_data['x'] = X.cpu().data.numpy().ravel()# if torch.cuda.is_available() else X.data.numpy().ravel()
            cloud_data['z'] = depth_in.cpu().data.numpy().ravel()# if torch.cuda.is_available() else depth_in.data.numpy().ravel()
            # semseg[i] = torch.from_numpy(np.load("data/check/00000_ss.npy")).to(self.gpu_device) #buat ngecek aja
            _, label_img = torch.max(semseg[i], dim=0)
            label_img = label_img.type(torch.int32)
            label_img = label_img.cpu().data.numpy()# if torch.cuda.is_available() else label_img.data.numpy()
            cloud_data['cls'] = label_img.ravel().view('<u4')
            # generated_sc = cloud_data.ravel()
            #yang melebihi batasan di buat nan
            cloud_data['x'][np.where(cloud_data['x']<-(self.cover_area/2))] = np.nan
            cloud_data['x'][np.where(cloud_data['x']>(self.cover_area/2))] = np.nan
            cloud_data['z'][np.where(cloud_data['z']>self.cover_area)] = np.nan
            #normalize ke frame 320x240
            cloud_data['x'] = np.round((cloud_data['x'] + (self.cover_area/2)) * (self.w-1) / self.cover_area)
            cloud_data['z'] = np.round((cloud_data['z'] * -(self.h-1) / self.cover_area) + (self.h-1))
            #map ke top view
            zipped_data = zip(cloud_data['x'], cloud_data['z'], cloud_data['cls'])
            onehot_arr = np.zeros((self.n_class, self.h, self.w))
            for x, z, cls in zipped_data:
                # print(x, z, cls)
                if np.isnan(x) or np.isnan(z):
                    continue
                else: #if x>=0 and x<self.w and z<self.h:
                    onehot_arr[cls][int(z)][int(x)] = 1
            # print(onehot_arr.shape)
            # np.save("00000_sc.npy", onehot_arr)
            sc_batch.append(onehot_arr)
        top_view_sc = torch.from_numpy(np.array(sc_batch)).float().to(self.gpu_device)
        return top_view_sc      
"""