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

# python crop_roi.py --units 5000 --data-dir "/media/sadevans/T7/ЛИЧНОЕ/Diplom/datsets/mTedx/ru" --detector retinaface --root-dir "/media/sadevans/T7/ЛИЧНОЕ/Diplom/datsets/mTedx/ru/train_data" --subset valid --dataset mtedx_ru --seg-duration 24

def preprocess_line(line):
    line = line.lower()
    line = re.sub(r'\([^(){}\[\]]*\)', '', line)
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
    parser.add_argument(
        "--units",
        type=int,
        default=5000,
        help="Index to identify separate jobs (useful for parallel processing).",
    )
    args = parser.parse_args()

    seg_duration = args.seg_duration
    dataset = args.dataset
    fps = 25

    SP_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "spm",
    "unigram_ru",
    f"unigram{args.units}_ru_vmeste.model",
    )

    DICT_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "spm",
        "unigram_ru",
        f"unigram{args.units}_units_vmeste.txt",
    )

    text_transform = TextTransform(sp_model_path=SP_MODEL_PATH, dict_path=DICT_PATH)

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
        f"{dataset}_{args.subset}_transcript_lengths_seg{seg_duration}s_{args.units}units.csv"
        if args.groups <= 1
        else f"{dataset}_{args.subset}_transcript_lengths_seg{seg_duration}s_{args.units}units.{args.groups}.{args.job_index}.csv",
    )

    another_label_filename = os.path.join(
        args.root_dir,
        "labels",
        f"{dataset}_{args.subset}_transcript_lengths_seg{seg_duration}s_{args.units}units_online.csv"
        if args.groups <= 1
        else f"{dataset}_{args.subset}_transcript_lengths_seg{seg_duration}s_{args.units}units_online.{args.groups}.{args.job_index}.csv",
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


    # if dataset.split('_')[0] == "mtedx":
    if args.subset in ["valid", "test", "train"]:
        # videos  = os.listdir(os.path.join(args.data_dir, args.subset, "video"))
        # sub_file  = os.path.join(args.data_dir, f"{args.subset}.txt")
        # filenames = []
        # for vid in videos:
        #     if vid == '.ipynb_checkpoints':
        #         continue
        #     else:
        # for el in os.listdir(os.path.join(args.data_dir, args.subset)):
        #     print(os.path.join(args.data_dir, args.subset, el))
        filenames = [os.path.join(args.data_dir, args.subset, el) \
                    for el in os.listdir(os.path.join(args.data_dir, args.subset))]
        # # random.shuffle(filenames)
        if args.subset in ["valid", "test"]: filenames.sort() 
        elif args.subset in ["train"]: random.shuffle(filenames)     
    else:
        raise NotImplementedError
        
    # filenames_processed = [os.path.join(args.data_dir, args.subset, video_filepath) for video_filepath in os.listdir(os.path.join(args.root_dir, args.dataset, f"{args.dataset}_video_seg{args.seg_duration}s"))]

    # filenames = list(set(filenames) - set(filenames_processed))

    print('LEN FILENAMES: ', len(filenames))

    unit = math.ceil(len(filenames) * 1.0 / args.groups)
    filenames = filenames[args.job_index * unit : (args.job_index + 1) * unit]

    sub_file  = open(os.path.join(args.data_dir, f"{args.subset}.txt"), "r").read().splitlines()
    # print(sub_file)
    print(len(sub_file))

    # for line in tqdm(sub_file):
    for line in tqdm(sub_file):

        data_filename = os.path.join(args.data_dir, args.subset, line.split(' ')[0] + '.mp4')
        # print(data_filename)
        # print(data_filename, os.path.exists(data_filename + '.mp4'))
        dst_vid_filename = os.path.join(dst_vid_dir, f"{line.split(' ')[0]}.mp4")
        dst_txt_filename = os.path.join(dst_txt_dir, f"{line.split(' ')[0]}.txt")
        # print(dst_vid_filename)
        if path.exists(data_filename):
            # print(data_filename)
            if not path.exists(dst_vid_filename):
            # print(dst_vid_filename)
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
                    # txt_data_filename = data_filename[:-4].replace('video', 'txt')
                    # text_line_list = (
                    #     open(txt_data_filename + ".txt", "r", encoding="utf-8").read().splitlines()[0].split(" ")
                    # )

                content = preprocess_line(" ".join(line.split(' ')[3:]))
                # print(data_filename, content)
                # break
                if video_data is None:
                    continue
                video_length = len(video_data)
                if video_length <= args.seg_duration * fps and video_length >= fps:
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
                        # f2 = open(another_label_filename, "a")

                        flag_open_labels = True
                        # flag_open_labels2 = True

                    
                    if flag_open_labels:
                        f.write(
                            "{}\n".format(
                                f"{dataset},{basename},{video_data.shape[0]},{token_id_str}"
                            )
                        )
                        f.close()
                        flag_open_labels = False

                    # if flag_open_labels2:
                    #     f2.write(
                    #         "{}\n".format(
                    #             f"{dataset},{basename},{video_data.shape[0]},{len(content)}"
                    #         )
                    #     )
                    #     f2.close()
                    #     flag_open_labels2 = False
                    torch.cuda.empty_cache()
                    gc.collect()
