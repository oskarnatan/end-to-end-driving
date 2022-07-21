import os
import json
import datetime
import pathlib
import time
import cv2
import carla
from collections import deque

from torch import torch

import torch
import carla
import numpy as np
from PIL import Image

from leaderboard.autoagents import autonomous_agent
from x8_B3.model import x8_B3
from x8_B3.config import GlobalConfig
from x8_B3.data import scale_and_crop_image, scale_and_crop_image_cv, rgb_to_depth, swap_RGB2BGR
from team_code.planner import RoutePlanner


SAVE_PATH = os.environ.get('SAVE_PATH', None)
CONTROL_OPTION = os.environ.get('CONTROL_OPTION', None)

def get_entry_point():
	return 'x8_B3Agent'


class x8_B3Agent(autonomous_agent.AutonomousAgent):
	def setup(self, path_to_conf_file):
		self.track = autonomous_agent.Track.SENSORS
		self.config_path = path_to_conf_file
		self.step = -1
		self.wall_start = time.time()
		self.initialized = False
		self.input_buffer = {'rgb': deque(), 'depth': deque(), 'gps': deque(), 'thetas': deque()} #'rgb_left': deque(), 'rgb_right': deque(), 'rgb_rear': deque(), 

		self.config = GlobalConfig()
		self.net = x8_B3(self.config, torch.device("cuda:0")).float().to(torch.device("cuda:0"))
		self.net.load_state_dict(torch.load(os.path.join(path_to_conf_file, 'best_model.pth')))
		self.net.cuda()
		self.net.eval()

		#control weights untuk PID dan MLP dari tuningan MGN
		#urutan steer, throttle, brake
		# self.cw = [[self.config.cw_pid[0], self.config.cw_pid[1], self.config.cw_pid[2]], [self.config.cw_mlp[0], self.config.cw_mlp[1], self.config.cw_mlp[2]]]

		self.save_path = None
		if SAVE_PATH is not None:
			now = datetime.datetime.now()
			string = pathlib.Path(os.environ['ROUTES']).stem + '_'
			string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
			self.sstring = string
			print (string)

			self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
			self.save_path.mkdir(parents=True, exist_ok=False)

			(self.save_path / 'rgb').mkdir()
			(self.save_path / 'depth').mkdir()
			(self.save_path / 'segmentation').mkdir()
			(self.save_path / 'semantic_cloud').mkdir()
			(self.save_path / 'meta').mkdir()

	def _init(self):
		self._route_planner = RoutePlanner(4.0, 50.0)
		self._route_planner.set_route(self._global_plan, True)

		self.initialized = True

	def _get_position(self, tick_data):
		gps = tick_data['gps']
		gps = (gps - self._route_planner.mean) * self._route_planner.scale

		return gps

	def sensors(self):
		return [
				{
					'type': 'sensor.camera.rgb',
					'x': 1.3, 'y': 0.0, 'z':2.3,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'width': 400, 'height': 300, 'fov': 100,
					'id': 'rgb'
					},
				{
					'type': 'sensor.camera.depth',
					'x': 1.3, 'y': 0.0, 'z':2.3,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'width': 400, 'height': 300, 'fov': 100,
					'id': 'depth'
					},
				# {
				# 	'type': 'sensor.camera.rgb',
				# 	'x': 1.3, 'y': 0.0, 'z':2.3,
				# 	'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
				# 	'width': 400, 'height': 300, 'fov': 100,
				# 	'id': 'rgb_left'
				# 	},
				# {
				# 	'type': 'sensor.camera.rgb',
				# 	'x': 1.3, 'y': 0.0, 'z':2.3,
				# 	'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
				# 	'width': 400, 'height': 300, 'fov': 100,
				# 	'id': 'rgb_right'
				# 	},
				# {
				# 	'type': 'sensor.camera.rgb',
				# 	'x': -1.3, 'y': 0.0, 'z':2.3,
				# 	'roll': 0.0, 'pitch': 0.0, 'yaw': -180.0,
				# 	'width': 400, 'height': 300, 'fov': 100,
				# 	'id': 'rgb_rear'
				# 	},
				{
					'type': 'sensor.other.imu',
					'x': 0.0, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.05,
					'id': 'imu'
					},
				{
					'type': 'sensor.other.gnss',
					'x': 0.0, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.01,
					'id': 'gps'
					},
				{
					'type': 'sensor.speedometer',
					'reading_frequency': 20,
					'id': 'speed'
					}
				]

	def tick(self, input_data):
		self.step += 1

		rgb = cv2.cvtColor(input_data['rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		depth = cv2.cvtColor(input_data['depth'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		# rgb_left = cv2.cvtColor(input_data['rgb_left'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		# rgb_right = cv2.cvtColor(input_data['rgb_right'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		# rgb_rear = cv2.cvtColor(input_data['rgb_rear'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		gps = input_data['gps'][1][:2]
		speed = input_data['speed'][1]['speed']
		compass = input_data['imu'][1][-1]

		result = {
				'rgb': rgb,
				'depth': depth,
				# 'rgb_left': rgb_left,
				# 'rgb_right': rgb_right,
				# 'rgb_rear': rgb_rear,
				'gps': gps,
				'speed': speed,
				'compass': compass,
				}
		
		pos = self._get_position(result)
		result['gps'] = pos
		next_wp, next_cmd = self._route_planner.run_step(pos)
		# result['next_command'] = next_cmd.value

		theta = compass + np.pi/2
		R = np.array([
			[np.cos(theta), -np.sin(theta)],
			[np.sin(theta), np.cos(theta)]
			])

		local_command_point = np.array([next_wp[0]-pos[0], next_wp[1]-pos[1]])
		local_command_point = R.T.dot(local_command_point)
		result['target_point'] = tuple(local_command_point)

		return result

	@torch.no_grad()
	def run_step(self, input_data, timestamp):
		if not self.initialized:
			self._init()

		tick_data = self.tick(input_data)

		if self.step < self.config.seq_len:
			rgb = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb']), scale=self.config.scale, crop=self.config.input_resolution)).unsqueeze(0)
			self.input_buffer['rgb'].append(rgb.to('cuda', dtype=torch.float32))
			
			# print(tick_data['depth'].shape)

			depth = torch.from_numpy(np.array(rgb_to_depth(scale_and_crop_image_cv(swap_RGB2BGR(tick_data['depth']), scale=self.config.scale, crop=self.config.input_resolution))))
			self.input_buffer['depth'].append(depth.to('cuda', dtype=torch.float32))
			"""
			if not self.config.ignore_sides:
				rgb_left = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_left']), scale=self.config.scale, crop=self.config.input_resolution)).unsqueeze(0)
				self.input_buffer['rgb_left'].append(rgb_left.to('cuda', dtype=torch.float32))
				
				rgb_right = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_right']), scale=self.config.scale, crop=self.config.input_resolution)).unsqueeze(0)
				self.input_buffer['rgb_right'].append(rgb_right.to('cuda', dtype=torch.float32))

			if not self.config.ignore_rear:
				rgb_rear = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_rear']), scale=self.config.scale, crop=self.config.input_resolution)).unsqueeze(0)
				self.input_buffer['rgb_rear'].append(rgb_rear.to('cuda', dtype=torch.float32))
			"""
			control = carla.VehicleControl()
			control.steer = 0.0
			control.throttle = 0.0
			control.brake = 0.0
			
			return control

		gt_velocity = torch.FloatTensor([tick_data['speed']]).to('cuda', dtype=torch.float32)
		# command = torch.FloatTensor([tick_data['next_command']]).to('cuda', dtype=torch.float32)

		tick_data['target_point'] = [torch.FloatTensor([tick_data['target_point'][0]]), torch.FloatTensor([tick_data['target_point'][1]])]
		target_point = torch.stack(tick_data['target_point'], dim=1).to('cuda', dtype=torch.float32)

		# encoding = []
		rgb = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb']), scale=self.config.scale, crop=self.config.input_resolution)).unsqueeze(0)
		self.input_buffer['rgb'].popleft()
		self.input_buffer['rgb'].append(rgb.to('cuda', dtype=torch.float32))
		# encoding.append(self.net.image_encoder(list(self.input_buffer['rgb'])))

		depth = torch.from_numpy(np.array(rgb_to_depth(scale_and_crop_image_cv(swap_RGB2BGR(tick_data['depth']), scale=self.config.scale, crop=self.config.input_resolution))))
		self.input_buffer['depth'].popleft()
		self.input_buffer['depth'].append(depth.to('cuda', dtype=torch.float32))
		
		"""
		if not self.config.ignore_sides:
			rgb_left = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_left']), scale=self.config.scale, crop=self.config.input_resolution)).unsqueeze(0)
			self.input_buffer['rgb_left'].popleft()
			self.input_buffer['rgb_left'].append(rgb_left.to('cuda', dtype=torch.float32))
			encoding.append(self.net.image_encoder(list(self.input_buffer['rgb_left'])))
			
			rgb_right = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_right']), scale=self.config.scale, crop=self.config.input_resolution)).unsqueeze(0)
			self.input_buffer['rgb_right'].popleft()
			self.input_buffer['rgb_right'].append(rgb_right.to('cuda', dtype=torch.float32))
			encoding.append(self.net.image_encoder(list(self.input_buffer['rgb_right'])))

		if not self.config.ignore_rear:
			rgb_rear = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_rear']), scale=self.config.scale, crop=self.config.input_resolution)).unsqueeze(0)
			self.input_buffer['rgb_rear'].popleft()
			self.input_buffer['rgb_rear'].append(rgb_rear.to('cuda', dtype=torch.float32))
			encoding.append(self.net.image_encoder(list(self.input_buffer['rgb_rear'])))
		"""

		# forward pass
		pred_seg, pred_wp, psteer, pthrottle, pbrake, pred_sc = self.net(self.input_buffer['rgb'], self.input_buffer['depth'], target_point, gt_velocity)
		mlp_steer = np.clip(psteer.cpu().data.numpy(), -1.0, 1.0)
		mlp_throttle = np.clip(pthrottle.cpu().data.numpy(), 0.0, self.config.max_throttle)
		mlp_brake = np.clip(pbrake.cpu().data.numpy(), 0.0, 1.0)

		# pid_steer, pid_throttle, pid_brake, pid_metadata = self.net.pid_control(pred_wp, gt_velocity) #PID ONLY
		steer, throttle, brake, metadata = self.net.mlp_pid_control(pred_wp, gt_velocity, mlp_steer, mlp_throttle, mlp_brake, CONTROL_OPTION) #MIX MLP AND PID
		# if brake < 0.05: brake = 0.0
		# if throttle > brake: brake = 0.0

		self.control_metadata = metadata
		#tambahan metadata, replace value yang ada di fungsi control model
		self.control_metadata['car_pos'] = tuple([float(tick_data['gps'][0]), float(tick_data['gps'][1])])
		self.control_metadata['next_point'] = tuple([float(tick_data['target_point'][0].cpu().data.numpy()), float(tick_data['target_point'][1].cpu().data.numpy())])


		control = carla.VehicleControl()
		control.steer = float(steer) #pid_steer mlp_steer steer
		control.throttle = float(throttle) #pid_throttle mlp_throttle throttle
		control.brake = float(brake) #pid_brake mlp_brake brake

		if SAVE_PATH is not None:# and self.step % 10 == 0:
			self.save(tick_data)
			self.save2(pred_seg, pred_sc)

		return control

	def save(self, tick_data):
		frame = self.step# // 10

		Image.fromarray(tick_data['rgb']).save(self.save_path / 'rgb' / ('%06d.png' % frame))
		Image.fromarray(swap_RGB2BGR(tick_data['depth'])).save(self.save_path / 'depth' / ('%06d.png' % frame))

		outfile = open(self.save_path / 'meta' / ('%06d.json' % frame), 'w')
		# print(self.control_metadata)
		json.dump(self.control_metadata, outfile, indent=4)
		outfile.close()
	
	def get_wp_nxr_frame(self):
		frame_dim = self.config.crop - 1 #array mulai dari 0
		area = self.config.coverage_area

		point_xy = []
		#proses wp
		""""""
		for i in range(1, self.config.pred_len+1):
			x_point = int((frame_dim/2) + (self.control_metadata['wp_'+str(i)][0]*(frame_dim/2)/area))
			y_point = int(frame_dim - (self.control_metadata['wp_'+str(i)][1]*frame_dim/area))
			xy_arr = np.clip(np.array([x_point, y_point]), 0, frame_dim)#constrain
			point_xy.append(xy_arr)
		
		#proses juga untuk next route
		# - + y point kebalikan dari WP, karena asumsinya agent mendekati next route point, dari negatif menuju 0
		x_point = int((frame_dim/2) + (self.control_metadata['next_point'][0]*(frame_dim/2)/area))
		y_point = int(frame_dim + (self.control_metadata['next_point'][1]*frame_dim/area))
		xy_arr = np.clip(np.array([x_point, y_point]), 0, frame_dim)#constrain
		point_xy.append(xy_arr)
		return point_xy

	def save2(self, ss, sc):
		frame = self.step# // 10
		ss = ss.cpu().detach().numpy()
		sc = sc.cpu().detach().numpy()

		#buat array untuk nyimpan out gambar
		imgx = np.zeros((ss.shape[2], ss.shape[3], 3))
		imgx2 = np.zeros((sc.shape[2], sc.shape[3], 3))
		#ambil tensor output segmentationnya
		pred_seg = ss[0]
		pred_sc = sc[0]
		inx = np.argmax(pred_seg, axis=0)
		inx2 = np.argmax(pred_sc, axis=0)
		for cmap in self.config.SEG_CLASSES['colors']:
			cmap_id = self.config.SEG_CLASSES['colors'].index(cmap)
			imgx[np.where(inx == cmap_id)] = cmap
			imgx2[np.where(inx2 == cmap_id)] = cmap
		# Image.fromarray(imgx).save(self.save_path / 'segmentation' / ('%06d.png' % frame))
		# Image.fromarray(imgx2).save(self.save_path / 'semantic_cloud' / ('%06d.png' % frame))
		
		#GANTI ORDER BGR KE RGB, SWAP!
		imgx = swap_RGB2BGR(imgx)
		imgx2 = swap_RGB2BGR(imgx2)

		wp_nxr_frame = self.get_wp_nxr_frame()
		""""""
		#gambar waypoints ke semantic cloud
		for i in range(self.config.pred_len):
			imgx2 = cv2.circle(imgx2, (wp_nxr_frame[i][0], wp_nxr_frame[i][1]), radius=2, color=(255, 255, 255), thickness=-1)
		
		#gambar juga next routenya, ada di element terakhir dalam list
		imgx2 = cv2.circle(imgx2, (wp_nxr_frame[-1][0], wp_nxr_frame[-1][1]), radius=4, color=(255, 255, 255), thickness=1)

		cwd = os.getcwd()
		# print(cwd+'/'+os.environ['SAVE_PATH']+'/'+self.sstring+'/segmentation/%06d.png' % frame)
		cv2.imwrite(cwd+'/'+os.environ['SAVE_PATH']+'/'+self.sstring+'/segmentation/%06d.png' % frame, imgx) #cetak predicted segmentation
		cv2.imwrite(cwd+'/'+os.environ['SAVE_PATH']+'/'+self.sstring+'/semantic_cloud/%06d.png' % frame, imgx2) #cetak predicted segmentation


	def destroy(self):
		del self.net
