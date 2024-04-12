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
warnings.filterwarnings("ignore")
    

def preprocess_line(line):
    line = line.upper()
    line = re.sub(r'^[^:]+:\s*', '', line)
    line = re.sub(r'\([^)]*\)', '', line)
    line = line.replace('Ё', 'Е')
    line = line.replace('ё', 'е')
    line = re.sub(r'[^0-9а-яА-Я- ]', '', line)
    line = line.replace('\t', '')
    line = re.sub(r'\s{2,}', ' ', line)

    return line


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
    parser.add_argument(
        "--subset",
        type=str,
        required=True,
        help="Subset of dataset - train, test, valid",
    )
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
    args = parser.parse_args()

    seg_duration = args.seg_duration
    dataset = args.dataset
    fps = 25
    text_transform = TextTransform()

    # Load Data
    args.data_dir = os.path.normpath(args.data_dir)

    # VIDEO DATALOADER
    vid_dataloader = AVSRDataLoader(
        modality="video", detector=args.detector, convert_gray=False
    )

    seg_vid_len = seg_duration * 25

    # Label filename
    label_filename = os.path.join(
        args.root_dir,
        "labels",
        f"{dataset}_{args.subset}_transcript_lengths_seg{seg_duration}s.csv"
        if args.groups <= 1
        else f"{dataset}_{args.subset}_transcript_lengths_seg{seg_duration}s.{args.groups}.{args.job_index}.csv",
    )

    another_label_filename = os.path.join(
        args.root_dir,
        "labels",
        f"{dataset}_{args.subset}_transcript_lengths_seg{seg_duration}s_online.csv"
        if args.groups <= 1
        else f"{dataset}_{args.subset}_transcript_lengths_seg{seg_duration}s_online.{args.groups}.{args.job_index}.csv",
    )

    # print('LABEL FILENAME: ', label_filename)

    os.makedirs(os.path.dirname(label_filename), exist_ok=True)
    print(f"Directory {os.path.dirname(label_filename)} created")

    if path.exists(label_filename):
        f = open(label_filename, 'a')
    else:
        f = open(label_filename, "w")

    if path.exists(another_label_filename):
        f2 = open(another_label_filename, 'a')
    else:
        f2 = open(another_label_filename, "w")
    flag_open_labels = True
    flag_open_labels2 = True


    # Step 2, extract mouth patches from segments.
    dst_vid_dir = os.path.join(
        args.root_dir, dataset, dataset + f"_video_seg{seg_duration}s"
    )
    dst_txt_dir = os.path.join(
        args.root_dir, dataset, dataset + f"_text_seg{seg_duration}s"
    )

    os.makedirs(dst_vid_dir, exist_ok=True)
    os.makedirs(dst_txt_dir, exist_ok=True)


    # print('DST VID DIR: ', dst_vid_dir)
    # print('DST TXT DIR: ', dst_txt_dir)

    # print('COMMON FILE: ', os.path.join(args.data_dir, args.subset, args.subset))

    if dataset.split('_')[0] == "mtedx":
        if args.subset in ["valid", "test"]:
            videos  = os.listdir(os.path.join(args.data_dir, args.subset, "video"))
            filenames = []
            for vid in videos:
                if vid == '.ipynb_checkpoints':
                    continue
                else:
                    filenames += [os.path.join(args.data_dir, args.subset, "video", vid, el) \
                                for el in os.listdir(os.path.join(args.data_dir, args.subset, "video", vid))]
                random.shuffle(filenames)
        elif args.subset == "train":
            videos  = os.listdir(os.path.join(args.data_dir, args.subset, "video"))
            filenames = []
            for vid in videos:
                # print('\nVID:',vid)
                if vid == '.ipynb_checkpoints':
                    continue
                else:
                    filenames += [os.path.join(args.data_dir, args.subset, "video", vid, el) \
                                for el in os.listdir(os.path.join(args.data_dir, args.subset, "video", vid))]
            filenames.sort()      
        else:
            raise NotImplementedError
        
    filenames_processed = [os.path.join(args.data_dir, args.subset, "video", video_filepath[:-9], video_filepath) for video_filepath in os.listdir(os.path.join(args.root_dir, args.dataset, f"{args.dataset}_video_seg{args.seg_duration}s"))]

    filenames = list(set(filenames) - set(filenames_processed))

    # print('FILENAMES: ', filenames)
    print('LEN FILENAMES: ', len(filenames))

    unit = math.ceil(len(filenames) * 1.0 / args.groups)
    filenames = filenames[args.job_index * unit : (args.job_index + 1) * unit]

    # i = 0
    for data_filename in tqdm(filenames):
        # print(os.path.normpath(data_filename).split(os.sep)[-4])

        dst_vid_filename = os.path.join(dst_vid_dir, f"{data_filename.split('/')[-1]}")
        dst_txt_filename = os.path.join(dst_txt_dir, f"{data_filename.split('/')[-1].replace('mp4', 'txt')}")

        if not path.exists(dst_vid_filename):
            if args.landmarks_dir:
                landmarks_filename = (
                    data_filename.replace(args.data_dir, args.landmarks_dir)[:-4] + ".pkl"
                )
                landmarks = pickle.load(open(landmarks_filename, "rb"))
            else:
                landmarks = None
            try:
                video_data = vid_dataloader.load_data(data_filename, landmarks)
            except (UnboundLocalError, TypeError, OverflowError, AssertionError):
                # print('here')
                continue

            txt_data_filename = data_filename[:-4].replace('video', 'txt')
            text_line_list = (
                open(txt_data_filename + ".txt", "r", encoding="utf-8").read().splitlines()[0].split(" ")
            )

            content = preprocess_line(" ".join(text_line_list[2:]))
            if video_data is None:
                continue
            video_length = len(video_data)
            if video_length <= args.seg_duration * fps:
                # print('SAVING...')
                save_vid_txt( 
                    dst_vid_filename,
                    dst_txt_filename,
                    video_data,
                    content,
                    video_fps=fps
                )
                # if i ==2: break
                basename = os.path.relpath(
                    dst_vid_filename, start=os.path.join(args.root_dir, dataset)
                )
                token_id_str = " ".join(
                    map(str, [_.item() for _ in text_transform.tokenize(content)])
                )
                if not flag_open_labels:
                    f = open(label_filename, "a")
                    f2 = open(another_label_filename, "a")

                    flag_open_labels = True
                    flag_open_labels2 = True

                
                if flag_open_labels:
                    f.write(
                        "{}\n".format(
                            f"{dataset},{basename},{video_data.shape[0]},{token_id_str}"
                        )
                    )
                    f.close()
                    flag_open_labels = False

                if flag_open_labels2:
                    f2.write(
                        "{}\n".format(
                            f"{dataset},{basename},{video_data.shape[0]},{len(content)}"
                        )
                    )
                    f2.close()
                    flag_open_labels2 = False
                torch.cuda.empty_cache()
                gc.collect()



                # continue
            # else:
            #     continue

        # print('SPLITTING....')
        # break
        # txt_data_filename = data_filename[:-4].replace('video', 'txt')
        # splitted = split_file(txt_data_filename + ".txt", max_frames=seg_vid_len)
        # for i in range(len(splitted)):
        #     if len(splitted) == 1:
        #         content, start, end, duration = splitted[i]
        #         trim_vid_data = video_data
        #     else:
        #         content, start, end, duration = splitted[i]
        #         print('CONTENT: ', content)
        #         start_idx, end_idx = int(start * 25), int(end * 25)
        #         try:
        #             trim_vid_data = (
        #                 video_data[start_idx:end_idx],
        #             )
        #         except TypeError:
        #             continue
        #     dst_vid_filename = os.path.join(dst_vid_dir, f"{data_filename.split('/')[-1]}")

        #     dst_txt_filename = os.path.join(dst_txt_dir, f"{data_filename.split('/')[-1].replace('mp4', 'txt')}")

        #     if trim_vid_data is None:
        #         continue
        #     video_length = len(trim_vid_data)
        #     if video_length == 0:
        #         continue
        #     save_vid_txt(
        #         dst_vid_filename,
        #         dst_txt_filename,
        #         trim_vid_data,
        #         content,
        #     )
        #     basename = os.path.relpath(
        #         dst_vid_filename, start=os.path.join(args.root_dir, dataset)
        #     )
        #     token_id_str = " ".join(
        #         map(str, [_.item() for _ in text_transform.tokenize(content)])
        #     )
        #     if token_id_str:
        #         if not flag_open_labels:
        #             f = open(label_filename, "a")
        #         else:
        #             flag_open_labels = False

                    
        #         f.write(
        #             "{}\n".format(
        #                 f"{dataset},{basename},{trim_vid_data.shape[0]},{token_id_str}"
        #             )
        #         )
        #         f.close()
        #         torch.cuda.empty_cache()
