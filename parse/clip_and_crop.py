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
from detectors.retinaface.detector import LandmarksDetector
import numpy as np
from moviepy.editor import VideoFileClip
import torchvision


warnings.filterwarnings("ignore")

# python crop_roi.py --units 5000 --data-dir "/media/sadevans/T7/ЛИЧНОЕ/Diplom/datsets/mTedx/ru" --detector retinaface --root-dir "/media/sadevans/T7/ЛИЧНОЕ/Diplom/datsets/mTedx/ru/train_data" --subset valid --dataset mtedx_ru --seg-duration 24

def milliseconds_to_seconds(milliseconds):
    seconds = milliseconds / 1000
    return seconds


def preprocess_line(line):
    line = line.lower()
    line = re.sub(r'\([^(){}\[\]]*\)', '', line)
    line = line.replace('Ё', 'Е')
    line = line.replace('ё', 'е')
    line = re.sub(r'[^0-9а-яА-Я- ]', '', line)
    line = line.replace('\t', '')
    line = re.sub(r'\s{2,}', ' ', line)

    return line


def preprocess_text(line):
    # line = line.upper()
    if '(Видео)' in line or '(видео)' in line:
        line = None
    elif bool(re.search(r'[a-zA-Z]', line)):
        line = None
    elif line.startswith('['):
        line = None
    else:
        line = line.lower()
        line = re.sub(r'\([^(){}\[\]]*\)', '', line)
        line = line.replace('Ё', 'Е')
        line = line.replace('ё', 'е')
        line = re.sub(r'[^0-9а-яА-Я- ]', '', line)
        line = line.replace('\t', '')
        line = re.sub(r'\s{2,}', ' ', line)
    if line is None or len(line) == 0:
        line = None

    return line


def load_video_text_data(data_folder, lang, group):
    """
    Parses YAML files found in the mTEDx dataset and returns video and text
    samples.

    Arguments
    ---------
    data_folder: str
        The absolute/relative path to the directory where the video file is
        located.
    lang : str
        The language code.
    group : list
        The group to be processed, e.g "test".
    """
    base_dir = (
        os.path.join(data_folder, f"{lang}-{lang}","data",group,"txt")
    )
    print(base_dir)
    # parse YAML file containing video (audio) info
    with open( os.path.join(base_dir, f"{group}.yaml"), "r") as fin:
        video_samples = yaml.load(fin, Loader=yaml.Loader)
    # parse text file containing text info

#     speaker_id_counts = defaultdict(int)

# # Подсчитываем количество строк с одинаковым speaker_id
#     for sample in video_samples:
#         speaker_id = sample.get('speaker_id')
#         speaker_id_counts[speaker_id] += 1

#     # Выводим результаты подсчёта
#     for speaker_id, count in speaker_id_counts.items():
#         print(f"Количество строк с speaker_id {speaker_id}: {count}")
    with open( os.path.join(base_dir, f"{group}.{lang}"), "r") as fin:
        text_samples = fin.readlines()
    # sanity check

    id_samples = list(np.arange(0, len(video_samples)))
    assert len(text_samples) == len(video_samples), \
        f"Data mismatch with language: {lang}, group: {group}"
    return video_samples, text_samples, id_samples



def split_and_save_video(input_path, output_path, start_time, end_time, seg_duration, content, landmarks_detector, vid_dataloader, input_fps, out_fps=25):
    """
    Splits the given video file into a segment based on the specified
    `start_time` & `end_time`. Then, it saves the videosegment in a mono wav
    format using `torchvision`.

    Arguments
    ---------
    input_path: str
        The absolute/relative path to the directory where the video file is
        located.
    output_path: str
        The absolute/relative path to the directory where the processed video
        file will be located.
    start_time: float
        The start time of the video segment.
    end_time: float
        The end time of the video segment.
    input_fps: int
        The input fps of the input video file.
    out_fps: int
        The output fps of the output video segment (default: 25).
    """
    # read the video file
    # video_frames, _, metadata = torchvision.io.read_video(input_path, start_pts=start_time, end_pts=end_time)

    # check faces
    video = VideoFileClip(str(input_path))
    start_time = milliseconds_to_seconds(start_time)
    end_time = milliseconds_to_seconds(end_time)
    try:
        segment = video.subclip(start_time, end_time)
        # segment.show()
        # print(segment.iter_frames())
        # success = False
        # landmarks_detector.fac
        landmarks = []
        frames = []
        for frame in segment.iter_frames():
            
            detected_faces = landmarks_detector.face_detector(frame, rgb=False)
            # print(detected_faces)
            if len(detected_faces) == 0:
                return False
            landmarks.append(detected_faces)
            frames.append(frames)
        del segment
        gc.collect()    
        # video_data = vid_dataloader.load_data(data_filename, landmarks)
        video_data = vid_dataloader.video_process(frames, landmarks)
        if video_data is None:
            return False
        video_length = len(video_data)
        if video_length <= seg_duration * fps and video_length >= OUT_FPS:
            save_vid_txt(output_path,
                        output_path.replace('video', 'text').replace('.mp4', 'txt'),
                        video_data,
                        content,
                        video_fps=fps
                    )
        
            # segment.write_videofile(str(output_path), fps=OUT_FPS, verbose=True)
            return True
    except:
        return False



def process_video_text_sample(i, video_data_dict, text, data_folder, dst_vid_dir, dst_txt_dir, lang, group, seg_duration):
    """
    Process one data sample.

    Arguments
    ---------
    i: int
        The index of the video file.
    video: dict
        A dictionary describing info about the video segment like:
        speaker_id, duration, offset, ... etc.
    text: str
        The text of the video segment.
    data_folder: str
        The absolute/relative path where the mTEDx data can be found.
    save_folder: str
        The absolute/relative path where the mTEDx data will be saved.
    
    Returns
    -------
    dict:
        A dictionary of audio-text segment info. 
    """

    text = preprocess_text(text)

    if text is None:
        return None
    
    video_input_filepath = (
        f"{data_folder}/{lang}-{lang}/data/{group}/video/{video_data_dict['speaker_id']}.mp4"
    )

    # file_count = sum(1 for file in os.listdir(dst_vid_dir) if file.startswith(f"{video['speaker_id']}_"))

    # print(file_count)

    video_segment_filename = video_data_dict["speaker_id"]+f"_{i:05d}"

    # video_output_filepath = (
    #     f"{dst_vid_dir}/{lang}/{group}/{video_segment_filename}.mp4"
    # )
    video_output_filepath = (
        f"{dst_vid_dir}/{video_segment_filename}.mp4"
    )

    text_output_filepath = (
        f"{dst_txt_dir}/{video_segment_filename}.txt"
    )

    if os.path.exists(video_output_filepath) or not os.path.exists(video_input_filepath):
        return None
    

    start_time = milliseconds_to_seconds(video_data_dict['offset'])
    end_time = milliseconds_to_seconds(video_data_dict['offset']+video_data_dict['duration'])
    # start_time = video_data_dict['offset']
    # end_time = video_data_dict['offset']+video_data_dict['duration']
    # print(start_time, end_time)
    # print(video_input_filepath, video_output_filepath)
    # video = VideoFileClip(video_input_filepath)
    # video = torchvision.io.read_video(video_input_filepath, pts_unit="sec")[0].numpy()
    video = torchvision.io.read_video(video_input_filepath, start_pts=start_time, end_pts=end_time, pts_unit="sec")[0].numpy()
    # print(video)
    try:
        # print('here')
        # video = VideoFileClip(video_input_filepath)
        # print(video)
        # segment = video.subclip(start_time, end_time)
        # print(segment)
        # landmarks = []
        # frames = []
        # for frame in video:
            
        #     detected_faces = landmarks_detector.face_detector(frame, rgb=False)
        #     print(detected_faces)
        #     if len(detected_faces) == 0:
        #         return None
        #     landmarks.append(detected_faces)
            # frames.append(frame)
        # del segment
        # gc.collect()    
        # video_data = vid_dataloader.load_data(data_filename, landmarks)
        # landmarks = landmarks_detector(video, rgb=False)
        landmarks = vid_dataloader.landmarks_detector(video)
        # print('in here')
        # print(landmarks)
        if landmarks is not None:
            video_data = vid_dataloader.video_process(video, landmarks)
            # print(video_data)
            if video_data is None:
                return None
        else: return None
        video_length = len(video_data)
        # print(video_length)
        # if video_length <= seg_duration * fps and video_length >= OUT_FPS:
        if video_length <= seg_duration * fps:

            save_vid_txt(video_output_filepath,
                        text_output_filepath,
                        video_data,
                        text,
                        video_fps=fps
                    )
            return video_output_filepath, video_data.shape[0], text
    except:
        return None
   

def preprocess(args, subset, lang):
    """
    Preprocess the mTEDx data found in the given language and the given group.
    Also, it writes the video-text information in a json file.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original mTEDx dataset is stored.
    save_folder : str
        Location of the folder for storing the csv.
    lang : str
        The language code.
    group : list
        The group to be processed, e.g "test".
    """
    label_filename = os.path.join(
        args.root_dir,
        "labels",
        f"{args.dataset}_{subset}_transcript_lengths_seg{args.seg_duration}s_{args.units}units.csv"
        if args.groups <= 1
        else f"{args.dataset}_{subset}_transcript_lengths_seg{args.seg_duration}s_{args.units}units.{args.groups}.{args.job_index}.csv",
    )
    print(label_filename)

    another_label_filename = os.path.join(
        args.root_dir,
        "labels",
        f"{args.dataset}_{subset}_transcript_lengths_seg{args.seg_duration}s_{args.units}units_online.csv"
        if args.groups <= 1
        else f"{args.dataset}_{subset}_transcript_lengths_seg{args.seg_duration}s_{args.units}units_online.{args.groups}.{args.job_index}.csv",
    )

    os.makedirs(os.path.dirname(label_filename), exist_ok=True)
    print(f"Directory {os.path.dirname(label_filename)} created")

    if path.exists(label_filename):
        f = open(label_filename, 'a')
        l = open(label_filename, 'r').readlines()
        print(f"File {label_filename} exists")

    else:
        f = open(label_filename, "w")
        l= []
        print(f"File {label_filename} created")

    # l = f.readlines()
    if path.exists(another_label_filename):
        f2 = open(another_label_filename, 'a')
    else:
        f2 = open(another_label_filename, "w")
    flag_open_labels = True
    flag_open_labels2 = True

    dst_vid_dir = os.path.join(
        args.root_dir, args.dataset, args.dataset + f"_video_seg{args.seg_duration}s"
    )
    dst_txt_dir = os.path.join(
        args.root_dir, args.dataset, args.dataset + f"_text_seg{args.seg_duration}s"
    )

    os.makedirs(dst_vid_dir, exist_ok=True)
    os.makedirs(dst_txt_dir, exist_ok=True)

    # print(args.data_dir, lang, subset)
    video_samples, text_samples, id_samples = load_video_text_data(args.data_dir, lang, subset)
    
    # processed_samples = 
    # i_samples = list(np.arange(0, len(video_samples)))
    if subset == 'train':
        print('Shuffle for train')
        zipped = list(zip(video_samples, text_samples, id_samples))
        random.seed(11)
        random.shuffle(zipped)
        video_samples, text_samples, id_samples = zip(*zipped)
    print(len(video_samples))
    if len(l) != 0:
        print(f'There are some lines in file: {len(l)}')
        last = int(l[-1].split(',')[1][-9:-4])
        print(last)
        print(id_samples.index(last))
        print(id_samples[last])
        print(id_samples[id_samples.index(last)])
        last_ind = id_samples.index(last)
        video_samples, text_samples, id_samples = video_samples[last_ind+1:], text_samples[last_ind+1:], id_samples[last_ind+1:]

    print(len(video_samples))
    for i, (video, text, id) in tqdm(enumerate(zip(video_samples, text_samples, id_samples))):
    # for i, (video, text, id) in enumerate(zip(video_samples, text_samples, id_samples)):
        # print(video)

        line = process_video_text_sample(id, video, text, args.data_dir, dst_vid_dir, dst_txt_dir, lang, subset, args.seg_duration)
        if line is not None:
            basename = os.path.relpath(line[0], start=os.path.join(args.root_dir, args.dataset))
            video_len, content = line[1], line[2]
            token_id_str = " ".join(map(str, [_.item() for _ in text_transform.tokenize(content)]))
            if not flag_open_labels:
                        f = open(label_filename, "a")
                        flag_open_labels = True

                    
            if flag_open_labels:
                f.write("{}\n".format(f"{args.dataset},{basename},{video_len},{token_id_str}"))
                f.close()
                flag_open_labels = False

            if flag_open_labels2:
                f2.write("{}\n".format(f"{args.dataset},{basename},{video_len},{len(content)}"))
                f2.close()
                flag_open_labels2 = False
            torch.cuda.empty_cache()
            gc.collect()



# def main(args):
#     for subset in args.subset:
#         for lang in args.langs:
#             preprocess(args.in_, args.root_dir, lang, args.subset, args.dataset, args.seg_duration, args.units, args.groups, args.job_index)



if __name__ == '__main__':
    # Argument Parsing
    parser = argparse.ArgumentParser(description="mTEDx Preprocessing")
    parser.add_argument(
        "--data-dir",
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
        "--root-dir",
        type=str,
        required=True,
        help="Root directory of preprocessed dataset",
    )
    # parser.add_argument(
    #     "--subset",
    #     type=str,
    #     required=True,
    #     help="Subset of dataset - train, test, valid",
    # )
    parser.add_argument('--subset', nargs='+', default="test valid train",
                help='List of groups separated by space, e.g. "valid train".')
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of dataset",
    )
    parser.add_argument(
        "--seg-duration",
        type=int,
        default=24,
        help="Max duration (second) for each segment, (Default: 24)",
    )
    parser.add_argument(
        "--combine-av",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Merges the audio and video components to a media file.",
    )
    parser.add_argument(
        "--groups",
        type=int,
        default=1,
        help="Number of threads to be used in parallel.",
    )
    parser.add_argument(
        "--job-index",
        type=int,
        default=0,
        help="Index to identify separate jobs (useful for parallel processing).",
    )
    parser.add_argument(
        "--units",
        type=int,
        default=5000,
        help="Index to identify separate jobs (useful for parallel processing).",
    )
    parser.add_argument('--langs', nargs='+', required=True,
                help='List of language codes separated by space, eg "de fr"')
    args = parser.parse_args()

    # seg_duration = args.seg_duration
    # dataset = args.dataset
    fps = 25
    # OUT_FPS = args["fps"]
    OUT_FPS = fps
    # SP_MODEL_PATH = os.path.join(
    # os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    # "spm",
    # "unigram_ru",
    # f"unigram{args.units}_ru.model",
    # )
    SP_MODEL_PATH = "/home/sadevans/space/preprocess_mtedx/spm/unigram_ru/unigram5000_ru_vmeste.model"

    # DICT_PATH = os.path.join(
    #     os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    #     "spm",
    #     "unigram_ru",
    #     f"unigram{args.units}_units.txt",
    # )
    DICT_PATH = "/home/sadevans/space/preprocess_mtedx/spm/unigram_ru/unigram5000_units_vmeste.txt"

    landmarks_detector = LandmarksDetector()
    vid_dataloader = AVSRDataLoader(modality="video", detector=args.detector, convert_gray=False)
    text_transform = TextTransform(sp_model_path=SP_MODEL_PATH, dict_path=DICT_PATH)

    args.data_dir = os.path.normpath(args.data_dir)

    for subset in args.subset:
        print(subset)
        for lang in args.langs:
            preprocess(args, subset, lang)
