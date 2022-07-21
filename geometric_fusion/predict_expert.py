import pandas as pd
import os
from tqdm import tqdm
from collections import OrderedDict
import time
import numpy as np
from torch import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True

from model import GeometricFusion
from data import CARLA_Data
from config import GlobalConfig
# import random
# random.seed(0)
# torch.manual_seed(0)


#Class untuk penyimpanan dan perhitungan update metric
class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    #update kalkulasi
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



#FUNGSI test
def test(data_loader, model, config, device):
    #buat variabel untuk menyimpan kalkulasi metric, dan iou
    score = {'wp_metric': AverageMeter()}

    #buat dictionary log untuk menyimpan training log di CSV
    log = OrderedDict([
        ('batch', []),
        ('test_wp_metric', []),
        ('elapsed_time', []),
    ])

    #buat save direktori
    save_dir = config.logdir + "/predict_expert/" + config.test_scenario 
    os.makedirs(save_dir, exist_ok=True)
            
    #masuk ke mode eval, pytorch
    model.eval()

    with torch.no_grad():
        #visualisasi progress validasi dengan tqdm
        prog_bar = tqdm(total=len(data_loader))

        #validasi....
        batch_ke = 1
        for data in data_loader:
            #load IO dan pindah ke GPU
            fronts = []
            lidars = []
            for i in range(config.seq_len): #append data untuk input sequence
                fronts.append(data['fronts'][i].to(device, dtype=torch.float))
                lidars.append(data['lidars'][i].to(device, dtype=torch.float))
            target_point = torch.stack(data['target_point'], dim=1).to(device, dtype=torch.float)
            gt_velocity = data['velocity'].to(device, dtype=torch.float)
            gt_waypoints = [torch.stack(data['waypoints'][i], dim=1).to(device, dtype=torch.float) for i in range(config.seq_len, len(data['waypoints']))]
            gt_waypoints = torch.stack(gt_waypoints, dim=1).to(device, dtype=torch.float)
            bev_points = data['bev_points'][0].long().to(device, dtype=torch.int64)
            cam_points = data['cam_points'][0].long().to(device, dtype=torch.int64)
            
            #forward pass
            start_time = time.time() #waktu mulai
            pred_wp = model(fronts, lidars, target_point, gt_velocity, bev_points, cam_points)
            elapsed_time = time.time() - start_time #hitung elapsedtime

            #compute metric
            metric_wp = F.l1_loss(pred_wp, gt_waypoints)
            
            #hitung rata-rata (avg) metric, dan metric untuk batch-batch yang telah diproses
            score['wp_metric'].update(metric_wp.item())

            #update visualisasi progress bar
            postfix = OrderedDict([('te_wp_m', score['wp_metric'].avg)])
            
            #simpan history training ke file csv
            log['batch'].append(batch_ke)
            log['test_wp_metric'].append(metric_wp.item())
            log['elapsed_time'].append(elapsed_time)
            #paste ke csv file
            pd.DataFrame(log).to_csv(os.path.join(save_dir, 'test_log_'+config.test_weather+'.csv'), index=False)

            batch_ke += 1  
            prog_bar.set_postfix(postfix)
            prog_bar.update(1)
        prog_bar.close()
        
        #ketika semua sudah selesai, hitung rata2 performa pada log
        log['batch'].append("avg")
        log['test_wp_metric'].append(np.mean(log['test_wp_metric']))
        log['elapsed_time'].append(np.mean(log['elapsed_time']))
        
        #ketika semua sudah selesai, hitung VARIANCE performa pada log
        log['batch'].append("stddev")
        log['test_wp_metric'].append(np.std(log['test_wp_metric'][:-1]))
        log['elapsed_time'].append(np.std(log['elapsed_time'][:-1]))

        #paste ke csv file
        pd.DataFrame(log).to_csv(os.path.join(save_dir, 'test_log_'+config.test_weather+'.csv'), index=False)


    #return value
    return log



# Load config
config = GlobalConfig()

#SET GPU YANG AKTIF
torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0")
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]=config.gpu_id#visible_gpu #"0" "1" "0,1"

#IMPORT MODEL dan load bobot
print("IMPORT ARSITEKTUR DL DAN COMPILE")
model = GeometricFusion(config, device).float().to(device)
model.load_state_dict(torch.load(os.path.join(config.logdir, 'best_model.pth')))

#BUAT DATA BATCH
test_set = CARLA_Data(root=config.test_data, config=config)
dataloader_test = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True) #BS selalu 1

#test
test_log = test(dataloader_test, model, config, device)



#kosongkan cuda chace
torch.cuda.empty_cache()

