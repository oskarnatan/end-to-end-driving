#!/bin/bash
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
export MODEL=transfuser #transfuser geometric_fusion late_fusion aim cilrs s13 x13
export CONTROL_OPTION=one_of #one_of both_must pid_only mlp_only, control option is only for s13 and x13 
export SAVE_PATH=data/ADVERSARIAL/${WEATHER}/${MODEL}_t2 #-${CONTROL_OPTION} # ADVERSARIAL NORMAL, Run1_ Run2_ Run3_, _w1 _t2 _t2w1 
export ROUTES=leaderboard/data/all_routes/routes_town02_long.xml #lihat di /leaderboard/data/all_routes
export SCENARIOS=leaderboard/data/scenarios/town02_all_scenarios.json #lihat di /leaderboard/data/scenarios town05_all_scenarios town02_all_scenarios no_scenarios.json
export PORT=2000 # same as the carla server port
export TM_PORT=2050 # port for traffic manager, required when spawning multiple servers/clients
export TEAM_CONFIG=${MODEL}/log/${MODEL}_t2 # _w1 _t2 _t2w1
export CHECKPOINT_ENDPOINT=${SAVE_PATH}/eval_result.json # results file
export TEAM_AGENT=leaderboard/team_code/${MODEL}_agent.py # agent: auto_pilot.py _agent.py

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT} \
--weather=${WEATHER} #tambahan buat ganti2 weather

