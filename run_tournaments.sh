export PYTHONPATH=./src
export CUDA_VISIBLE_DEVICES=""

cat games | grep -v '^#' | grep . | while read game; do
  echo $game
  if [ "$1" == "background" ]; then
    python src/models/run_tournament.py $game 10000 &
  else
    python src/models/run_tournament.py $game 2000
  fi
done
