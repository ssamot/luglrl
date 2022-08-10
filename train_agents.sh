export PYTHONPATH=./src
export CUDA_VISIBLE_DEVICES=""


cat agents | grep -v '^#' | grep . | while read agent
do
    cat games | grep -v '^#' | grep . | while read game
    do
      echo $game $agent
      mkdir models/games/$game
      if [ "$1" == "background" ]
      then
        python src/models/train_agents.py $game $agent 5000 1000001 2>logs/$game\_$agent.err >logs/$game\_$agent.out &
      else
        python src/models/train_agents.py $game $agent 5000 200001
      fi
    done
done
