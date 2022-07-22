export PYTHONPATH=./src
export CUDA_VISIBLE_DEVICES=""

cat games | grep -v '^#' | grep . | while read game; do
  echo $game
  if [ "$1" == "background" ]; then
    python src/visualization/calculate_glicko2_scores.py data/interim/$game.csv &
  else
    python src/visualization/calculate_glicko2_scores.py data/interim/$game.csv
  fi
done
