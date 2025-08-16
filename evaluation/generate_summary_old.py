# -*- coding: utf-8 -*-
import numpy as np
from knapsack_implementation import knapSack


def generate_summary(all_shot_bound, all_scores, all_number_of_shots):
    """ Generate the automatic machine summary, based on the video shots; the frame importance scores; the number of
    frames in the original video and the position of the sub-sampled frames of the original video.

    :param list[np.ndarray] all_shot_bound: The video shots for all the -original- testing videos.
    :param list[np.ndarray] all_scores: The calculated frame importance scores for all the sub-sampled testing videos.
    :param list[np.ndarray] all_nframes: The number of frames for all the -original- testing videos.
    :param list[np.ndarray] all_positions: The position of the sub-sampled frames for all the -original- testing videos.
    :return: A list containing the indices of the selected frames for all the -original- testing videos.
    """
    all_summaries = []
    for video_index in range(len(all_scores)):
        shot_scores = all_scores[video_index]
        list_of_shot_scores = []
        shape = shot_scores.shape
        for x in shot_scores:
            for y in x:
                list_of_shot_scores.append(y)
        shot_bound = all_shot_bound[video_index]
        num_of_shot = all_number_of_shots[video_index]
        shot_lengths = []
        for shot in shot_bound:
            shot_lengths.append(shot[1] - shot[0] + 1)
        final_shot = shot_bound[-1]
        final_max_length = int((final_shot[1] + 1) * 0.15)

        selected = knapSack(final_max_length, shot_lengths, list_of_shot_scores, len(shot_lengths))

        # Select all frames from each selected shot (by setting their value in the summary vector to 1)
        summary = np.zeros(final_shot[1] + 1, dtype=np.int8)
        for shot in selected:
            summary[shot_bound[shot][0]:shot_bound[shot][1] + 1] = 1

        all_summaries.append(summary)

    return all_summaries
