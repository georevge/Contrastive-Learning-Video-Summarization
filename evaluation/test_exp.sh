# -*- coding: utf-8 -*-
# Bash script to automate the procedure of evaluating an experiment.
# The script is taking as granted that the experiments are saved under "base_path" as "base_path/exp$EXP_NUM".
# First, -for each split- get the training loss from tensorboard as csv file. Then, compute the fscore (txt file)
# for the current experiment. Finally, based ONLY on the training loss choose the best epoch (and model).
base_path="/home/gevge/Downloads/PGL-SUM-CAP-Embedding_space/Summaries"
exp_num=$1
dataset=$2
eval_method=$3  # SumMe -> max | TVSum avg

exp_path="$base_path/exp$exp_num"; echo $exp_path

for i in 0 1 2 3 4; do
  results_path="$exp_path/$dataset/results/split$i"
  python evaluation/compute_fscores_test.py --path $results_path --dataset $dataset --eval $eval_method
done
