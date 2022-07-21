# End-to-end Autonomous Driving with Semantic Depth Cloud Mapping and Multi-agent

O. Natan and J. Miura, “End-to-end Autonomous Driving with Semantic Depth Cloud Mapping and Multi-agent,” IEEE Trans. Intelligent Vehicles, 2022. [[paper]](https://doi.org/10.1109/TIV.2022.3185303) 

## Related works:
1. O. Natan and J. Miura, “DeepIPC: Deeply Integrated Perception and Control for Mobile Robot in Real Environments,” arXiv:2207.09934, 2022. [[paper]](https://arxiv.org/abs/2207.09934) 
2. O. Natan and J. Miura, “Towards Compact Autonomous Driving Perception with Balanced Learning and Multi-sensor Fusion,” IEEE Trans. Intelligent Transportation Systems, 2022. [[paper]](https://doi.org/10.1109/TITS.2022.3149370) [[code]](https://github.com/oskarnatan/compact-perception)
3. O. Natan and J. Miura, "Semantic Segmentation and Depth Estimation with RGB and DVS Sensor Fusion for Multi-view Driving Perception," in Proc. Asian Conf. Pattern Recognition (ACPR), Jeju Island, South Korea, Nov. 2021, pp. 352–365. [[paper]](https://doi.org/10.1007/978-3-031-02375-0_26) [[code]](https://github.com/oskarnatan/RGBDVS-fusion)

## Notes:
1. Some files are copied and modified from [[TransFuser, CVPR 2021]](https://github.com/autonomousvision/transfuser) repository.

## Steps:
1. Download the dataset and extract to subfolder data. Or generate the data by yourself.
2. To train-val-test each model, go to their folder and read the instruction written in the README.md file

## Generate Data and Automated Driving Evaluation:
1. Run CARLA server:
    - Go to https://github.com/carla-simulator/carla/releases/tag/0.9.10.1
    - Download Prebuilt CARLA and its additional maps
    - Extract to a directory (e.g., ~/OSKAR/CARLA/CARLA_0.9.10.1)
    - Run CARLA Server, CUDA_VISIBLE_DEVICES=0 ~/OSKAR/CARLA/CARLA_0.9.10/CarlaUE4.sh -opengl --world-port=2000
4. To generate data / collect data, Run expert (results are saved in subfolder 'data'):
  - CUDA_VISIBLE_DEVICES=0 ./leaderboard/scripts/run_expert.sh
3. For automated driving, Run agents (results are saved in subfolder 'data'):
  - CUDA_VISIBLE_DEVICES=0 ./leaderboard/scripts/run_evaluation.sh

## To do list:
1. Add download link for the dataset
