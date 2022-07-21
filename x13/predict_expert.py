import pandas as pd
import os
from tqdm import tqdm
from collections import OrderedDict
import time
import numpy as np
from torch import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True

from model import x13
from data import CARLA_Data
from config import GlobalConfig


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def IOU(Yp, Yt):
    output = Yp.view(-1) > 0.5 
    target = Yt.view(-1) > 0.5 
    intersection = (output & target).sum() 
    union = (output | target).sum()
    iou = intersection / union
    return iou



def test(data_loader, model, config, device):
    score = {'total_metric': AverageMeter(),
            'ss_metric': AverageMeter(),
            'wp_metric': AverageMeter(),
            'str_metric': AverageMeter(),
            'thr_metric': AverageMeter(),
            'brk_metric': AverageMeter(),
            'redl_metric': AverageMeter(),
            'stops_metric': AverageMeter()}

    log = OrderedDict([
        ('batch', []),
        ('test_metric', []),
        ('test_ss_metric', []),
        ('test_wp_metric', []),
        ('test_str_metric', []),
        ('test_thr_metric', []),
        ('test_brk_metric', []),
        ('test_redl_metric', []),
        ('test_stops_metric', []),
        ('elapsed_time', []),
    ])

    save_dir = config.logdir + "/predict_expert/" + config.test_scenario 
    os.makedirs(save_dir, exist_ok=True)
            
    model.eval()

    with torch.no_grad():
        prog_bar = tqdm(total=len(data_loader))

        #validasi....
        batch_ke = 1
        for data in data_loader:
            fronts = data['fronts'].to(device, dtype=torch.float) 
            seg_fronts = data['seg_fronts'].to(device, dtype=torch.float) 
            depth_fronts = data['depth_fronts'].to(device, dtype=torch.float)
            target_point = torch.stack(data['target_point'], dim=1).to(device, dtype=torch.float)
            gt_velocity = data['velocity'].to(device, dtype=torch.float)
            gt_waypoints = [torch.stack(data['waypoints'][i], dim=1).to(device, dtype=torch.float) for i in range(config.seq_len, len(data['waypoints']))]
            gt_waypoints = torch.stack(gt_waypoints, dim=1).to(device, dtype=torch.float)
            gt_steer = data['steer'].to(device, dtype=torch.float)
            gt_throttle = data['throttle'].to(device, dtype=torch.float)
            gt_brake = data['brake'].to(device, dtype=torch.float)
            gt_red_light = data['red_light'].to(device, dtype=torch.float)
            gt_stop_sign = data['stop_sign'].to(device, dtype=torch.float)

            #forward pass
            start_time = time.time() # mulai
            pred_seg, pred_wp, steer, throttle, brake, red_light, stop_sign, _ = model(fronts, depth_fronts, target_point, gt_velocity)
            elapsed_time = time.time() - start_time # elapsedtime

            #compute metric
            metric_seg = IOU(pred_seg, seg_fronts)
            metric_wp = F.l1_loss(pred_wp, gt_waypoints)
            metric_str = F.l1_loss(steer, gt_steer)
            metric_thr = F.l1_loss(throttle, gt_throttle)
            metric_brk = F.l1_loss(brake, gt_brake)
            loss_redl = F.l1_loss(red_light, gt_red_light)
            metric_redl = 1 - torch.round(loss_redl)
            loss_stops = F.l1_loss(stop_sign, gt_stop_sign)
            metric_stops = 1 - torch.round(loss_stops)
            total_metric = (1-metric_seg) + metric_wp + metric_str + metric_thr + metric_brk + (1-metric_redl) + (1-metric_stops)

            score['total_metric'].update(total_metric.item())
            score['ss_metric'].update(metric_seg.item()) 
            score['wp_metric'].update(metric_wp.item())
            score['str_metric'].update(metric_str.item())
            score['thr_metric'].update(metric_thr.item())
            score['brk_metric'].update(metric_brk.item())
            score['redl_metric'].update(metric_redl.item())
            score['stops_metric'].update(metric_stops.item())

            postfix = OrderedDict([('te_total_m', score['total_metric'].avg),
                                ('te_ss_m', score['ss_metric'].avg),
                                ('te_wp_m', score['wp_metric'].avg),
                                ('te_str_m', score['str_metric'].avg),
                                ('te_thr_m', score['thr_metric'].avg),
                                ('te_brk_m', score['brk_metric'].avg),
                                ('te_redl_m', score['redl_metric'].avg),
                                ('te_stops_m', score['stops_metric'].avg)])
            
            log['batch'].append(batch_ke)
            log['test_metric'].append(total_metric.item())
            log['test_ss_metric'].append(metric_seg.item())
            log['test_wp_metric'].append(metric_wp.item())
            log['test_str_metric'].append(metric_str.item())
            log['test_thr_metric'].append(metric_thr.item())
            log['test_brk_metric'].append(metric_brk.item())
            log['test_redl_metric'].append(metric_redl.item())
            log['test_stops_metric'].append(metric_stops.item())
            log['elapsed_time'].append(elapsed_time)
            pd.DataFrame(log).to_csv(os.path.join(save_dir, 'test_log_'+config.test_weather+'.csv'), index=False)

            batch_ke += 1  
            prog_bar.set_postfix(postfix)
            prog_bar.update(1)
        prog_bar.close()
        
        log['batch'].append("avg")
        log['test_metric'].append(np.mean(log['test_metric']))
        log['test_ss_metric'].append(np.mean(log['test_ss_metric']))
        log['test_wp_metric'].append(np.mean(log['test_wp_metric']))
        log['test_str_metric'].append(np.mean(log['test_str_metric']))
        log['test_thr_metric'].append(np.mean(log['test_thr_metric']))
        log['test_brk_metric'].append(np.mean(log['test_brk_metric']))
        log['test_redl_metric'].append(np.mean(log['test_redl_metric']))
        log['test_stops_metric'].append(np.mean(log['test_stops_metric']))
        log['elapsed_time'].append(np.mean(log['elapsed_time']))
        
        log['batch'].append("stddev")
        log['test_metric'].append(np.std(log['test_metric'][:-1]))
        log['test_ss_metric'].append(np.std(log['test_ss_metric'][:-1]))
        log['test_wp_metric'].append(np.std(log['test_wp_metric'][:-1]))
        log['test_str_metric'].append(np.std(log['test_str_metric'][:-1]))
        log['test_thr_metric'].append(np.std(log['test_thr_metric'][:-1]))
        log['test_brk_metric'].append(np.std(log['test_brk_metric'][:-1]))
        log['test_redl_metric'].append(np.std(log['test_redl_metric'][:-1]))
        log['test_stops_metric'].append(np.std(log['test_stops_metric'][:-1]))
        log['elapsed_time'].append(np.std(log['elapsed_time'][:-1]))

        pd.DataFrame(log).to_csv(os.path.join(save_dir, 'test_log_'+config.test_weather+'.csv'), index=False)


    #return value
    return log



# Load config
config = GlobalConfig()

#SET GPU
torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0")
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]=config.gpu_id#visible_gpu #"0" "1" "0,1"

#IMPORT MODEL
print("IMPORT ARSITEKTUR DL DAN COMPILE")
model = x13(config, device).float().to(device)
model.load_state_dict(torch.load(os.path.join(config.logdir, 'best_model.pth')))

#DATA BATCH
test_set = CARLA_Data(root=config.test_data, config=config)
dataloader_test = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True) #BS 1

#test
test_log = test(dataloader_test, model, config, device)


#kosongkan cuda chace
torch.cuda.empty_cache()

