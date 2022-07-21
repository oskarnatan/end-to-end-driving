#!/bin/bash
export CARLA_ROOT=~/OSKAR/Transfuser/xtransfuser/carla/CARLA_0.9.10.1
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner

#original
export LEADERBOARD_ROOT=leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
export RESUME=True



#tambahan n edit
export WEATHER=ClearNoon # ClearNoon, ClearSunset, CloudyNoon, CloudySunset, WetNoon, WetSunset, MidRainyNoon, MidRainSunset, WetCloudyNoon, WetCloudySunset, HardRainNoon, HardRainSunset, SoftRainNoon, SoftRainSunset
export SAVE_PATH=data/NORMAL2_2/Run1_${WEATHER}/Expert # ADVERSARIAL NORMAL, Run1_ Run2_ Run3_, _w1 _t2 _t2w1 
export ROUTES=leaderboard/data/all_routes/routes_town02_long.xml #lihat di /leaderboard/data/all_routes
export SCENARIOS=leaderboard/data/scenarios/no_scenarios.json #lihat di /leaderboard/data/scenarios town05_all_scenarios town02_all_scenarios no_scenarios.json
export PORT=11000 # same as the carla server port
export TM_PORT=11050 # port for traffic manager, required when spawning multiple servers/clients
export CHECKPOINT_ENDPOINT=${SAVE_PATH}/eval_result.json # results file
export TEAM_AGENT=leaderboard/team_code/expert.py

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT} \
--weather=${WEATHER} #tambahan buat ganti2 weather

