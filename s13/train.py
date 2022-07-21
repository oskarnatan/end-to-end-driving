import pandas as pd
import os
from tqdm import tqdm
from collections import OrderedDict
import time
import numpy as np
from torch import torch, nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True

from model import s13
from data import CARLA_Data
from config import GlobalConfig
from torch.utils.tensorboard import SummaryWriter
# import random
# random.seed(0)
# torch.manual_seed(0)


#Class untuk penyimpanan dan perhitungan update loss
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

#Class NN Module untuk Perhitungan BCE Dice Loss
def BCEDice(Yp, Yt, smooth=1e-7):
	#.view(-1) artinya matrix tensornya di flatten kan dulu
	Yp = Yp.view(-1)
	Yt = Yt.view(-1)
	#hitung BCE
	bce = F.binary_cross_entropy(Yp, Yt, reduction='mean')
	#hitung dice loss
	intersection = (Yp * Yt).sum() #irisan
	#rumus DICE
	dice_loss = 1 - ((2. * intersection + smooth) / (Yp.sum() + Yt.sum() + smooth))
	#kalkulasi lossnya
	bce_dice_loss = bce + dice_loss
	return bce_dice_loss


# def weighted_bce(Yp, Yt, weights=[1, 1.5]): #weights untuk class 0 dan class 1
# 	Yt = Yt.view(-1)
# 	Yp = torch.clamp(Yp.view(-1), min=1e-7, max=1-1e-7)
# 	loss = -1 * torch.mean(weights[1]*Yt*torch.log(Yp) + weights[0]*(1-Yt)*torch.log(1-Yp))
# 	return loss


#fungsi renormalize loss weights seperti di paper gradnorm
def renormalize_params_lw(current_lw, config):
	#detach dulu paramsnya dari torch, pindah ke CPU
	lw = np.array([tens.cpu().detach().numpy() for tens in current_lw])
	lws = np.array([lw[i][0] for i in range(len(lw))])
	#fungsi renormalize untuk algoritma 1 di papaer gradnorm
	coef = np.array(config.loss_weights).sum()/lws.sum()
	new_lws = [coef*lwx for lwx in lws]
	#buat torch float tensor lagi dan masukkan ke cuda memory
	normalized_lws = [torch.cuda.FloatTensor([lw]).clone().detach().requires_grad_(True) for lw in new_lws]
	return normalized_lws

#FUNGSI TRAINING
def train(data_loader, model, config, writer, cur_epoch, device, optimizer, params_lw, optimizer_lw):
	#buat variabel untuk menyimpan kalkulasi loss, dan iou
	score = {'total_loss': AverageMeter(),
			'ss_loss': AverageMeter(),
			'wp_loss': AverageMeter(),
			'str_loss': AverageMeter(),
			'thr_loss': AverageMeter(),
			'brk_loss': AverageMeter(),
			'redl_loss': AverageMeter(),
			'stops_loss': AverageMeter()}
	
	#masuk ke mode training, pytorch
	model.train()

	#visualisasi progress training dengan tqdm
	prog_bar = tqdm(total=len(data_loader))

	#training....
	total_batch = len(data_loader)
	batch_ke = 0
	for data in data_loader:
		cur_step = cur_epoch*total_batch + batch_ke

		#load IO dan pindah ke GPU
		# fronts = []
		# for i in range(config.seq_len): #append data untuk input sequence
		# 	fronts.append(data['fronts'][i].to(device, dtype=torch.float))
		fronts = data['fronts'].to(device, dtype=torch.float) #ambil yang terakhir aja #[-1]
		seg_fronts = data['seg_fronts'].to(device, dtype=torch.float) #ambil yang terakhir aja #[-1]
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
		pred_seg, pred_wp, steer, throttle, brake, red_light, stop_sign = model(fronts, target_point, gt_velocity)#, seg_fronts[-1])

		#compute loss
		loss_seg = BCEDice(pred_seg, seg_fronts)
		loss_wp = F.l1_loss(pred_wp, gt_waypoints)
		loss_str = F.l1_loss(steer, gt_steer)
		loss_thr = F.l1_loss(throttle, gt_throttle)
		loss_brk = F.l1_loss(brake, gt_brake)
		loss_redl = F.l1_loss(red_light, gt_red_light)
		loss_stops = F.l1_loss(stop_sign, gt_stop_sign)
		total_loss = params_lw[0]*loss_seg + params_lw[1]*loss_wp + params_lw[2]*loss_str + params_lw[3]*loss_thr + params_lw[4]*loss_brk + params_lw[5]*loss_redl + params_lw[6]*loss_stops

		#backpro, kalkulasi gradient, dan optimasi
		optimizer.zero_grad()

		if batch_ke == 0: #batch pertama, hitung loss awal
			total_loss.backward() #ga usah retain graph
			#ambil loss pertama
			loss_seg_0 = torch.clone(loss_seg)
			loss_wp_0 = torch.clone(loss_wp)
			loss_str_0 = torch.clone(loss_str)
			loss_thr_0 = torch.clone(loss_thr)
			loss_brk_0 = torch.clone(loss_brk)
			loss_redl_0 = torch.clone(loss_redl)
			loss_stops_0 = torch.clone(loss_stops)

		elif 0 < batch_ke < total_batch-1:
			total_loss.backward() #ga usah retain graph

		elif batch_ke == total_batch-1: #berarti batch terakhir, compute update loss weights
			if config.MGN:
				optimizer_lw.zero_grad()
				total_loss.backward(retain_graph=True) #backpro, hitung gradient, retain graph karena graphnya masih dipakai perhitungan
				#ambil nilai gradient dari layer pertama pada masing2 task-specified decoder dan komputasi gradient dari output layer sampai ke bottle neck saja
				params = list(filter(lambda p: p.requires_grad, model.parameters()))
				G0R = torch.autograd.grad(loss_seg, params[config.bottleneck[0]], retain_graph=True, create_graph=True)
				G0 = torch.norm(G0R[0], keepdim=True).squeeze()
				G1R = torch.autograd.grad(loss_wp, params[config.bottleneck[1]], retain_graph=True, create_graph=True)
				G1 = torch.norm(G1R[0], keepdim=True).squeeze()
				G2R = torch.autograd.grad(loss_str, params[config.bottleneck[1]], retain_graph=True, create_graph=True)
				G2 = torch.norm(G2R[0], keepdim=True).squeeze()
				G3R = torch.autograd.grad(loss_thr, params[config.bottleneck[1]], retain_graph=True, create_graph=True)
				G3 = torch.norm(G3R[0], keepdim=True).squeeze()
				G4R = torch.autograd.grad(loss_brk, params[config.bottleneck[1]], retain_graph=True, create_graph=True)
				G4 = torch.norm(G4R[0], keepdim=True).squeeze()
				G5R = torch.autograd.grad(loss_redl, params[config.bottleneck[0]], retain_graph=True, create_graph=True)
				G5 = torch.norm(G5R[0], keepdim=True).squeeze()
				G6R = torch.autograd.grad(loss_stops, params[config.bottleneck[0]], retain_graph=True, create_graph=True)
				G6 = torch.norm(G6R[0], keepdim=True).squeeze()
				#dan rata2
				G_avg = (G0+G1+G2+G3+G4+G5+G6) / len(config.loss_weights)

				#hitung relative lossnya
				loss_seg_hat = loss_seg / loss_seg_0
				loss_wp_hat = loss_wp / loss_wp_0
				loss_str_hat = loss_str / loss_str_0
				loss_thr_hat = loss_thr / loss_thr_0
				loss_brk_hat = loss_brk / loss_brk_0
				loss_redl_hat = loss_redl / loss_redl_0
				loss_stops_hat = loss_stops / loss_stops_0
				#dan rata2
				loss_hat_avg = (loss_seg_hat + loss_wp_hat + loss_str_hat + loss_thr_hat + loss_brk_hat + loss_redl_hat + loss_stops_hat) / len(config.loss_weights)

				#hitung r_i_(t) relative inverse training rate untuk setiap task 
				inv_rate_ss = loss_seg_hat / loss_hat_avg
				inv_rate_wp = loss_wp_hat / loss_hat_avg
				inv_rate_str = loss_str_hat / loss_hat_avg
				inv_rate_thr = loss_thr_hat / loss_hat_avg
				inv_rate_brk = loss_brk_hat / loss_hat_avg
				inv_rate_redl = loss_redl_hat / loss_hat_avg
				inv_rate_stops = loss_stops_hat / loss_hat_avg

				#hitung constant target grad
				C0 = (G_avg*inv_rate_ss).squeeze().detach()**config.lw_alpha
				C1 = (G_avg*inv_rate_wp).squeeze().detach()**config.lw_alpha
				C2 = (G_avg*inv_rate_str).squeeze().detach()**config.lw_alpha
				C3 = (G_avg*inv_rate_thr).squeeze().detach()**config.lw_alpha
				C4 = (G_avg*inv_rate_brk).squeeze().detach()**config.lw_alpha
				C5 = (G_avg*inv_rate_redl).squeeze().detach()**config.lw_alpha
				C6 = (G_avg*inv_rate_stops).squeeze().detach()**config.lw_alpha

				#HITUNG TOTAL LGRAD
				Lgrad = F.l1_loss(G0, C0) + F.l1_loss(G1, C1) + F.l1_loss(G2, C2) + F.l1_loss(G3, C3) + F.l1_loss(G4, C4) + F.l1_loss(G5, C5) + F.l1_loss(G6, C6)

				#hitung gradient loss sesuai Eq. 2 di GradNorm paper
				# optimizer_lw.zero_grad()
				Lgrad.backward()
				#update loss weights
				optimizer_lw.step() 

				#ambil lgrad untuk disimpan nantinya
				lgrad = Lgrad.item()
				new_param_lw = optimizer_lw.param_groups[0]['params']
				# print(new_param_lw)
			else:
				total_loss.backward()
				lgrad = 0
				new_param_lw = 1
			
		optimizer.step() #dan update bobot2 pada network model

		#hitung rata-rata (avg) loss, dan metric untuk batch-batch yang telah diproses
		score['total_loss'].update(total_loss.item())
		score['ss_loss'].update(loss_seg.item()) 
		score['wp_loss'].update(loss_wp.item())
		score['str_loss'].update(loss_str.item())
		score['thr_loss'].update(loss_thr.item())
		score['brk_loss'].update(loss_brk.item())
		score['redl_loss'].update(loss_redl.item())
		score['stops_loss'].update(loss_stops.item())

		#update visualisasi progress bar
		postfix = OrderedDict([('t_total_l', score['total_loss'].avg),
							('t_ss_l', score['ss_loss'].avg),
							('t_wp_l', score['wp_loss'].avg),
							('t_str_l', score['str_loss'].avg),
							('t_thr_l', score['thr_loss'].avg),
							('t_brk_l', score['brk_loss'].avg),
							('t_redl_l', score['redl_loss'].avg),
							('t_stops_l', score['stops_loss'].avg)])
		
		#tambahkan ke summary writer
		writer.add_scalar('t_total_l', total_loss.item(), cur_step)
		writer.add_scalar('t_ss_l', loss_seg.item(), cur_step)
		writer.add_scalar('t_wp_l', loss_wp.item(), cur_step)
		writer.add_scalar('t_str_l', loss_str.item(), cur_step)
		writer.add_scalar('t_thr_l', loss_thr.item(), cur_step)
		writer.add_scalar('t_brk_l', loss_brk.item(), cur_step)
		writer.add_scalar('t_redl_l', loss_redl.item(), cur_step)
		writer.add_scalar('t_stops_l', loss_stops.item(), cur_step)

		prog_bar.set_postfix(postfix)
		prog_bar.update(1)
		batch_ke += 1
	prog_bar.close()	

	#return value
	return postfix, new_param_lw, lgrad


#FUNGSI VALIDATION
def validate(data_loader, model, config, writer, cur_epoch, device):
	#buat variabel untuk menyimpan kalkulasi loss, dan iou
	score = {'total_loss': AverageMeter(),
			'ss_loss': AverageMeter(),
			'wp_loss': AverageMeter(),
			'str_loss': AverageMeter(),
			'thr_loss': AverageMeter(),
			'brk_loss': AverageMeter(),
			'redl_loss': AverageMeter(),
			'stops_loss': AverageMeter()}
			
	#masuk ke mode eval, pytorch
	model.eval()

	with torch.no_grad():
		#visualisasi progress validasi dengan tqdm
		prog_bar = tqdm(total=len(data_loader))

		#validasi....
		total_batch = len(data_loader)
		batch_ke = 0
		for data in data_loader:
			cur_step = cur_epoch*total_batch + batch_ke

			#load IO dan pindah ke GPU
			# fronts = []
			# for i in range(config.seq_len): #append data untuk input sequence
			# 	fronts.append(data['fronts'][i].to(device, dtype=torch.float))
			fronts = data['fronts'].to(device, dtype=torch.float) #ambil yang terakhir aja #[-1]
			seg_fronts = data['seg_fronts'].to(device, dtype=torch.float) #ambil yang terakhir aja #[-1]
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
			pred_seg, pred_wp, steer, throttle, brake, red_light, stop_sign = model(fronts, target_point, gt_velocity)#, seg_fronts[-1])

			#compute loss
			loss_seg = BCEDice(pred_seg, seg_fronts)
			loss_wp = F.l1_loss(pred_wp, gt_waypoints)
			loss_str = F.l1_loss(steer, gt_steer)
			loss_thr = F.l1_loss(throttle, gt_throttle)
			loss_brk = F.l1_loss(brake, gt_brake)
			loss_redl = F.l1_loss(red_light, gt_red_light)
			loss_stops = F.l1_loss(stop_sign, gt_stop_sign)
			total_loss = loss_seg + loss_wp + loss_str + loss_thr + loss_brk + loss_redl + loss_stops

			#hitung rata-rata (avg) loss, dan metric untuk batch-batch yang telah diproses
			score['total_loss'].update(total_loss.item())
			score['ss_loss'].update(loss_seg.item()) 
			score['wp_loss'].update(loss_wp.item())
			score['str_loss'].update(loss_str.item())
			score['thr_loss'].update(loss_thr.item())
			score['brk_loss'].update(loss_brk.item())
			score['redl_loss'].update(loss_redl.item())
			score['stops_loss'].update(loss_stops.item())

			#update visualisasi progress bar
			postfix = OrderedDict([('v_total_l', score['total_loss'].avg),
								('v_ss_l', score['ss_loss'].avg),
								('v_wp_l', score['wp_loss'].avg),
								('v_str_l', score['str_loss'].avg),
								('v_thr_l', score['thr_loss'].avg),
								('v_brk_l', score['brk_loss'].avg),
								('v_redl_l', score['redl_loss'].avg),
								('v_stops_l', score['stops_loss'].avg)])
			
			#tambahkan ke summary writer
			writer.add_scalar('v_total_l', total_loss.item(), cur_step)
			writer.add_scalar('v_ss_l', loss_seg.item(), cur_step)
			writer.add_scalar('v_wp_l', loss_wp.item(), cur_step)
			writer.add_scalar('v_str_l', loss_str.item(), cur_step)
			writer.add_scalar('v_thr_l', loss_thr.item(), cur_step)
			writer.add_scalar('v_brk_l', loss_brk.item(), cur_step)
			writer.add_scalar('v_redl_l', loss_redl.item(), cur_step)
			writer.add_scalar('v_stops_l', loss_stops.item(), cur_step)

			prog_bar.set_postfix(postfix)
			prog_bar.update(1)
			batch_ke += 1
		prog_bar.close()	

	#return value
	return postfix


#MAIN FUNCTION
def main():
	# Load config
	config = GlobalConfig()

	#SET GPU YANG AKTIF
	torch.backends.cudnn.benchmark = True
	device = torch.device("cuda:0")
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
	os.environ["CUDA_VISIBLE_DEVICES"]=config.gpu_id#visible_gpu #"0" "1" "0,1"

	#IMPORT MODEL UNTUK DITRAIN
	print("IMPORT ARSITEKTUR DL DAN COMPILE")
	model = s13(config, device).float().to(device)
	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print('Total trainable parameters: ', params)

	#KONFIGURASI OPTIMIZER
	# optima = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)
	optima = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optima, mode='min', factor=0.5, patience=3, min_lr=1e-5)

	#BUAT DATA BATCH
	train_set = CARLA_Data(root=config.train_data, config=config)
	val_set = CARLA_Data(root=config.val_data, config=config)
	# print(len(train_set))
	if len(train_set)%config.batch_size == 1:
		drop_last = True #supaya ga mengacaukan MGN #drop last perlu untuk MGN
	else: #selain 1 bisa
		drop_last = False
	dataloader_train = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=drop_last) 
	dataloader_val = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
	# print(len(dataloader_train))
	
	#cek retrain atau tidak
	if not os.path.exists(config.logdir+"/trainval_log.csv"):
		print('TRAIN from the beginning!!!!!!!!!!!!!!!!')
		os.makedirs(config.logdir, exist_ok=True)
		print('Created dir:', config.logdir)
		#optimizer lw
		params_lw = [torch.cuda.FloatTensor([config.loss_weights[i]]).clone().detach().requires_grad_(True) for i in range(len(config.loss_weights))]
		optima_lw = optim.SGD(params_lw, lr=config.lr)
		#set nilai awal
		curr_ep = 0
		lowest_score = float('inf')
		stop_count = config.init_stop_counter
	else:
		print('Continue training!!!!!!!!!!!!!!!!')
		print('Loading checkpoint from ' + config.logdir)
		#baca log history training sebelumnya
		log_trainval = pd.read_csv(config.logdir+"/trainval_log.csv")
		# replace variable2 ini
		# print(log_trainval['epoch'][-1:])
		curr_ep = int(log_trainval['epoch'][-1:]) + 1
		lowest_score = float(np.min(log_trainval['val_loss']))
		stop_count = int(log_trainval['stop_counter'][-1:])
		# Load checkpoint
		model.load_state_dict(torch.load(os.path.join(config.logdir, 'recent_model.pth')))
		optima.load_state_dict(torch.load(os.path.join(config.logdir, 'recent_optim.pth')))

		#set optima lw baru
		latest_lw = [float(log_trainval['lw_ss'][-1:]), float(log_trainval['lw_wp'][-1:]), float(log_trainval['lw_str'][-1:]), float(log_trainval['lw_thr'][-1:]), float(log_trainval['lw_brk'][-1:]), float(log_trainval['lw_redl'][-1:]), float(log_trainval['lw_stops'][-1:])]
		params_lw = [torch.cuda.FloatTensor([latest_lw[i]]).clone().detach().requires_grad_(True) for i in range(len(latest_lw))]
		optima_lw = optim.SGD(params_lw, lr=float(log_trainval['lrate'][-1:]))
		# optima_lw.param_groups[0]['lr'] = optima.param_groups[0]['lr'] # lr disamakan
		# optima_lw.load_state_dict(torch.load(os.path.join(config.logdir, 'recent_optim_lw.pth')))
		#update direktori dan buat tempat penyimpanan baru
		config.logdir += "/retrain"
		os.makedirs(config.logdir, exist_ok=True)
		print('Created new retrain dir:', config.logdir)

	#buat dictionary log untuk menyimpan training log di CSV
	log = OrderedDict([
			('epoch', []),
			('best_model', []),
			('val_loss', []),
			('val_ss_loss', []),
			('val_wp_loss', []),
			('val_str_loss', []),
			('val_thr_loss', []),
			('val_brk_loss', []),
			('val_redl_loss', []),
			('val_stops_loss', []),
			('train_loss', []), 
			('train_ss_loss', []),
			('train_wp_loss', []),
			('train_str_loss', []),
			('train_thr_loss', []),
			('train_brk_loss', []),
			('train_redl_loss', []),
			('train_stops_loss', []),
			('lrate', []),
			('stop_counter', []), 
			('lgrad_loss', []),
			('lw_ss', []),
			('lw_wp', []),
			('lw_str', []),
			('lw_thr', []),
			('lw_brk', []),
			('lw_redl', []),
			('lw_stops', []),
			('elapsed_time', []),
		])
	writer = SummaryWriter(log_dir=config.logdir)
	
	#proses iterasi tiap epoch
	epoch = curr_ep
	while True:
		print("Epoch: {:05d}------------------------------------------------".format(epoch))
		#cetak lr dan lw
		if config.MGN:
			curr_lw = optima_lw.param_groups[0]['params']
			lw = np.array([tens.cpu().detach().numpy() for tens in curr_lw])
			lws = np.array([lw[i][0] for i in range(len(lw))])
			print("current loss weights: ", lws)	
		else:
			curr_lw = config.loss_weights
			lws = config.loss_weights
			print("current loss weights: ", config.loss_weights)
		print("current lr untuk training: ", optima.param_groups[0]['lr'])

		#training validation
		start_time = time.time() #waktu mulai
		train_log, new_params_lw, lgrad = train(dataloader_train, model, config, writer, epoch, device, optima, curr_lw, optima_lw)
		val_log = validate(dataloader_val, model, config, writer, epoch, device)
		if config.MGN:
			#update params lw yang sudah di renormalisasi ke optima_lw
			optima_lw.param_groups[0]['params'] = renormalize_params_lw(new_params_lw, config) #harus diclone supaya benar2 terpisah
			print("total loss gradient: "+str(lgrad))
		#update learning rate untuk training process
		scheduler.step(val_log['v_total_l']) #parameter acuan reduce LR adalah val_total_metric
		optima_lw.param_groups[0]['lr'] = optima.param_groups[0]['lr'] #update lr disamakan
		elapsed_time = time.time() - start_time #hitung elapsedtime

		#simpan history training ke file csv
		log['epoch'].append(epoch)
		log['lrate'].append(optima.param_groups[0]['lr'])
		log['train_loss'].append(train_log['t_total_l'])
		log['val_loss'].append(val_log['v_total_l'])
		log['train_ss_loss'].append(train_log['t_ss_l'])
		log['val_ss_loss'].append(val_log['v_ss_l'])
		log['train_wp_loss'].append(train_log['t_wp_l'])
		log['val_wp_loss'].append(val_log['v_wp_l'])
		log['train_str_loss'].append(train_log['t_str_l'])
		log['val_str_loss'].append(val_log['v_str_l'])
		log['train_thr_loss'].append(train_log['t_thr_l'])
		log['val_thr_loss'].append(val_log['v_thr_l'])
		log['train_brk_loss'].append(train_log['t_brk_l'])
		log['val_brk_loss'].append(val_log['v_brk_l'])
		log['train_redl_loss'].append(train_log['t_redl_l'])
		log['val_redl_loss'].append(val_log['v_redl_l'])
		log['train_stops_loss'].append(train_log['t_stops_l'])
		log['val_stops_loss'].append(val_log['v_stops_l'])
		log['lgrad_loss'].append(lgrad)
		log['lw_ss'].append(lws[0])
		log['lw_wp'].append(lws[1])
		log['lw_str'].append(lws[2])
		log['lw_thr'].append(lws[3])
		log['lw_brk'].append(lws[4])
		log['lw_redl'].append(lws[5])
		log['lw_stops'].append(lws[6])
		log['elapsed_time'].append(elapsed_time)
		print('| t_total_l: %.4f | t_ss_l: %.4f | t_wp_l: %.4f | t_str_l: %.4f | t_thr_l: %.4f | t_brk_l: %.4f | t_redl_l: %.4f | t_stops_l: %.4f |' % (train_log['t_total_l'], train_log['t_ss_l'], train_log['t_wp_l'], train_log['t_str_l'], train_log['t_thr_l'], train_log['t_brk_l'], train_log['t_redl_l'], train_log['t_stops_l']))
		print('| v_total_l: %.4f | v_ss_l: %.4f | v_wp_l: %.4f | v_str_l: %.4f | v_thr_l: %.4f | v_brk_l: %.4f | v_redl_l: %.4f | v_stops_l: %.4f |' % (val_log['v_total_l'], val_log['v_ss_l'], val_log['v_wp_l'], val_log['v_str_l'], val_log['v_thr_l'], val_log['v_brk_l'], val_log['v_redl_l'], val_log['v_stops_l']))
		print('elapsed time: %.4f sec' % (elapsed_time))
		
		#save recent model dan optimizernya
		torch.save(model.state_dict(), os.path.join(config.logdir, 'recent_model.pth'))
		torch.save(optima.state_dict(), os.path.join(config.logdir, 'recent_optim.pth'))
		# torch.save(optima_lw.state_dict(), os.path.join(config.logdir, 'recent_optim_lw.pth'))

		#save model best only
		if val_log['v_total_l'] < lowest_score:
			print("v_total_l: %.4f < lowest sebelumnya: %.4f" % (val_log['v_total_l'], lowest_score))
			print("model terbaik disave!")
			torch.save(model.state_dict(), os.path.join(config.logdir, 'best_model.pth'))
			torch.save(optima.state_dict(), os.path.join(config.logdir, 'best_optim.pth'))
			# torch.save(optima_lw.state_dict(), os.path.join(config.logdir, 'best_optim_lw.pth'))
			#v_total_l sekarang menjadi lowest_score
			lowest_score = val_log['v_total_l']
			#reset stop counter
			stop_count = config.init_stop_counter
			print("stop counter direset ke: ", stop_count)
			#catat sebagai best model
			log['best_model'].append("BEST")
		else:
			print("v_total_l: %.4f >= lowest sebelumnya: %.4f" % (val_log['v_total_l'], lowest_score))
			print("model tidak disave!")
			stop_count -= 1
			print("stop counter : ", stop_count)
			log['best_model'].append("")

		#update stop counter
		log['stop_counter'].append(stop_count)
		#paste ke csv file
		pd.DataFrame(log).to_csv(os.path.join(config.logdir, 'trainval_log.csv'), index=False)

		#kosongkan cuda chace
		torch.cuda.empty_cache()
		epoch += 1

		# early stopping jika stop counter sudah mencapai 0 dan early stop true
		if stop_count==0:
			print("TRAINING BERHENTI KARENA TIDAK ADA PENURUNAN TOTAL LOSS DALAM %d EPOCH TERAKHIR" % (config.init_stop_counter))
			break #loop
		

#RUN PROGRAM
if __name__ == "__main__":
	main()


