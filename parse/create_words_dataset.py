import argparse
import math
import os
from os import path
from pathlib import Path
import pickle
import warnings
from data.data_module import AVSRDataLoader
from tqdm import tqdm
from transforms import TextTransform
from utils import save_vid_aud_txt, split_file, save_vid_txt
import torch
import random
import gc
import string
import re
import yaml
from collections import defaultdict


import numpy as np
from moviepy.editor import VideoFileClip
import torchvision

import yaml


def filter_words_in_yaml(speakers_count, yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        data = yaml.safe_load(file)
        filtered_data = {key: value for key, value in data.items() if int(value) >= speakers_count}
        return filtered_data


def get_timestamps_speakers(words_dict, yaml_speakers_file):
    with open(yaml_speakers_file, 'r') as file:
        data = yaml.safe_load(file)
    words_dict_set = set(list(words_dict.keys()))
    data_set = set(list(data.keys()))
    needed_keys = list(words_dict_set.intersection(data_set))
    filtered_keys = {key: data[key] for key in needed_keys}

    return filtered_keys


def save2vid(filename, vid, frames_per_second):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torchvision.io.write_video(filename, vid, frames_per_second)


def clip_speaker_video(args, i, word, speaker, start, end):
    # duration = end - start
    # print(duration)
    # if duration <= 1.6:
    #     ost = 1.6 - duration
    #     end = end - ost//2
    #     start = start + (ost - ost//2)
    print("START, END: ", start, end)
    clip_video_frames = read_video(args.data_dir, speaker, start, end)
    print(clip_video_frames.shape)
    # if args.detector == "mediapipe":
    #     landmarks = landmarks_detector(clip_video_frames)
    #     if 
    #     cropped_video = video_process(clip_video_frames, landmarks)

    # else:
    landmarks = landmarks_detector(clip_video_frames)
    if landmarks is not None:
        cropped_video = video_process(clip_video_frames, landmarks)


        filename = f"{word}_{speaker}_{i:05d}.mp4"
        
        os.makedirs(os.path.join(args.root_dir, word), exist_ok=True)
        path = os.path.join(args.root_dir, word)
        # print(cropped_video.shape)
        if cropped_video is not None:
            save2vid(os.path.join(path, filename), cropped_video, 25)



def read_video(path, speaker, start, end):
    video_frames = torchvision.io.read_video(os.path.join(path, speaker + '.mp4'), pts_unit='sec', start_pts=start, end_pts=end)[0].numpy()
    return video_frames


def make_words_videos(words_speakers_dict, args):
    for word, speakers_data in words_speakers_dict.items():
        # print(word)
        os.makedirs(os.path.join(args.root_dir, word), exist_ok=True)

        speakers = np.unique(np.array([speaker['speaker_id'] for speaker in speakers_data]))
        random_speakers = np.random.choice(speakers, size=100, replace=False)
        processed_speakers = []

        for i, speaker in enumerate(speakers_data):
            # print("in cycle")
            if speaker['speaker_id'] in random_speakers and speaker['speaker_id'] not in processed_speakers:
        #     # print(speaker['end'] - speaker['start'])
                # print("here")
                filename = f"{word}_{speaker['speaker_id']}_{i:05d}.mp4"
                path = os.path.join(args.root_dir, word)
                if os.path.exists(os.path.join(path, filename)):
                    continue

                duration = speaker['end'] - speaker['start']
                if duration <= 1.6:
                    ost = 1.6 - duration
                    if speaker['start'] > ost/2:
                        start = speaker['start'] - (ost/2)
                        end = speaker['end'] + (ost - ost/2)

                    else:
                        end = speaker['end'] + ost
                        start = speaker['start']

                    clip_speaker_video(args, i, word, speaker['speaker_id'], start, end)
                    processed_speakers.append(speaker['speaker_id'])

                elif duration > 1.6:
                    continue
            





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mTEDx Preprocessing")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory of original dataset",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="retinaface",
        help="Type of face detector. (Default: retinaface)",
    )
    parser.add_argument(
        "--landmarks-dir",
        type=str,
        default=None,
        help="Directory of landmarks",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory of preprocessed dataset",
    )
    # parser.add_argument(
    #     "--dataset",
    #     type=str,
    #     required=True,
    #     help="Name of dataset",
    # )
    parser.add_argument(
        "--num_speakers",
        type=int,
        required=True,
        help="Name of dataset",
    )
    
    # top100_words = ['вид', 'вопрос', '']
    
    args = parser.parse_args()
    random.seed(11)

    if args.detector == "retinaface":
        from detectors.retinaface.detector import LandmarksDetector
        from detectors.retinaface.video_process import VideoProcess
        landmarks_detector = LandmarksDetector()
        # vid_dataloader = AVSRDataLoader(modality="video", detector=args.detector, convert_gray=False)
        video_process = VideoProcess(resize=(256, 256))
    elif args.detector == "mediapipe":
        from detectors.mediapipe.detector import LandmarksDetector
        from detectors.mediapipe.video_process import VideoProcess
        landmarks_detector = LandmarksDetector()
        # vid_dataloader = AVSRDataLoader(modality="video", detector=args.detector, convert_gray=False)
        video_process = VideoProcess(convert_gray=False, crop_width=96,crop_height=96)

    # Пример использования:
    speakers_count = args.num_speakers
    yaml_file_path = 'word_num_speaker.yaml'  # Замените на путь к вашему файлу
    yaml_speakers_file = 'unique_words_and_speakers.yaml'
    
    filtered_yaml_data = filter_words_in_yaml(speakers_count, yaml_file_path) # слова, у которым минимум speakers_count спикеров различных
    # print(filtered_yaml_data)

    filtered_words_speakers_data = get_timestamps_speakers(filtered_yaml_data, yaml_speakers_file)
    # print(filtered_words_speakers_data)
    make_words_videos(filtered_words_speakers_data, args)
