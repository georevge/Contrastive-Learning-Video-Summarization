# -*- coding: utf-8 -*-
from os import listdir
import json
import numpy as np
import h5py
from evaluation_metrics import evaluate_summary
from generate_summary import generate_summary
import argparse
import shutil

# arguments to run the script
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str,
                    default='...',
                    help="Path to the json files with the scores of the frames for each epoch")
parser.add_argument("--model_path", type=str,
                    default='...',
                    help="Path to the json files with the scores of the frames for each epoch")
parser.add_argument("--dataset", type=str, default='SumMe', help="Dataset to be used")
parser.add_argument("--eval", type=str, default="max", help="Eval method to be used for f_score reduction (max or avg)")

args = vars(parser.parse_args())
path = args["path"]
model_path = args["model_path"]
dataset = args["dataset"]
eval_method = args["eval"]

results = [f for f in listdir(path) if f.endswith(".json") and not f.endswith("test.json")]
results.sort(key=lambda video: int(video[6:-5]))
dataset_path = '/home/gevge/Downloads/PGL-SUM-CAP-Embedding_space/data/' + dataset + '/eccv16_dataset_' + dataset.lower() + '_google_pool5.h5'

f_score_epochs = []
epoch_number = -1
for epoch in results:                       # for each epoch ...
    all_scores = []
    all_user_summary, all_shot_bound, all_nframes, all_positions, all_number_of_shots = [], [], [], [], []
    with open(path + '/' + epoch) as f:     # read the json file ...
        data = json.loads(f.read())
        keys = list(data.keys())

    with h5py.File(dataset_path, 'r') as hdf:
        for video_name in keys:
            video_index = video_name[6:]

            user_summary = np.array(hdf.get('video_' + video_index + '/user_summary'))
            sb = np.array(hdf.get('video_' + video_index + '/change_points'))
            number_of_shots = len(sb)
            n_frames = np.array(hdf.get('video_' + video_index + '/n_frames'))
            positions = np.array(hdf.get('video_' + video_index + '/picks'))

            all_user_summary.append(user_summary)
            all_shot_bound.append(sb)
            all_number_of_shots.append(number_of_shots)
            all_nframes.append(n_frames)
            all_positions.append(positions)

        i = 0
        for video_name in keys:             # for each video inside that json file ...
            scores = np.asarray(data[video_name])  #.squeeze(0)  # read the importance scores from frames  # edw den xreiazetai to squeeze
            # scores = np.repeat(scores, 4)

            # scores = scores[:all_nframes[i]]
            all_scores.append(scores)
            # i = i+1

    all_summaries = generate_summary(all_shot_bound, all_scores, all_nframes, all_positions)

    all_f_scores = []
    # compare the resulting summary with the ground truth one, for each video
    for video_index in range(len(all_summaries)):
        summary = all_summaries[video_index]
        user_summary = all_user_summary[video_index]
        f_score = evaluate_summary(summary, user_summary, eval_method)
        all_f_scores.append(f_score)

    f_score_epochs.append(np.mean(all_f_scores))
    print("Epoch:", epoch_number, " f_score: ", np.mean(all_f_scores))  # begin from epoch -1 (untrained)
    epoch_number = epoch_number + 1
# Save the importance scores in txt format.
print("Max f1-score : ", max(f_score_epochs[1:]), " epoch: ", f_score_epochs.index(max(f_score_epochs[1:])) - 1)  # begin from epoch 0
with open(path + '/f_scores.txt', 'w') as outfile:
    json.dump(f_score_epochs, outfile)

with open(path + '/max_f_score.txt', 'w') as outfile2:
    outfile2.write(("Max f1-score : " + str(max(f_score_epochs[1:])) + ", epoch: " + str(f_score_epochs.index(max(f_score_epochs[1:])) - 1)))  # f_score_epochs -> list [0,300]
n = f_score_epochs.index(max(f_score_epochs[1:])) - 1
shutil.copy(model_path + '/epoch-' + str(n) + '.pkl', model_path + '/best_model.pkl')
