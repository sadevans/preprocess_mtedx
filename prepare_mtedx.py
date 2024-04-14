import numpy as np
import datetime
import os
import argparse
from pathlib import Path
from collections import defaultdict, OrderedDict
import gc
import warnings
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import ffmpeg
import tarfile
from os import path
import wget
import bisect
import re
import string


# SPLITS = ['train', 'test', 'valid']
SPLITS = ['valid', 'test']
# SPLITS = ['train']

def time_to_seconds(time_str):
    """
        Convert time in format %H:%M:%S.%f to %S.f
        Arguments:
            - time_str - a string with default time
        Return:
            - total_seconds - float seconds
    """
    time_str = time_str.replace('\n', '')
    time_str = time_str.replace(' ', '')

    time_obj = datetime.datetime.strptime(time_str, "%H:%M:%S.%f")
    total_seconds = time_obj.second + time_obj.minute * 60 + time_obj.hour * 3600 + time_obj.microsecond / 1000000
    return total_seconds


def preprocess_line(line):
    line = line.upper()
    line = re.sub(r'\([^(){}\[\]]*\)', '', line)
    line = line.replace('Ё', 'Е')
    line = line.replace('ё', 'е')
    line = re.sub(r'[^0-9а-яА-Я- ]', '', line)
    line = line.replace('\t', '')
    line = re.sub(r'\s{2,}', ' ', line)

    return line


def read_txt_file(txt_filepath:Path):
    """
        Read text files
        Arguments:
            - txt_filepath- the path to your txt file
        Return:
            all lines from your txt file
    """
    with open(txt_filepath) as fin:
        return [line.strip() for line in fin.readlines()]


# def download_mtedx_data(download_path, src, tgt):
#     """Downloads mTEDx data from OpenSLR"""
#     tgz_filename = f"mtedx_{src}-{tgt}.tgz" if src != tgt else f"mtedx_{src}.tgz"
#     download_extract_file_if_not(
#         url=f"https://www.openslr.org/resources/100/{tgz_filename}",
#         tgz_filepath=download_path / tgz_filename,
#         download_filename=f"{src}-{tgt}"
#     )



def preprocess_vtt_files(mtedx_path: Path, src_lang: str, duration_threshold: int) -> None:
    """
        Make transcriptions file for video segments for all videos in all splits.
        Arguments:
            - mtedx_path - the path to mtedx dataset
            - src_lang - source language
            - duration_threshold - maximum length of a video segment in seconds
    """
    for split in SPLITS:
        # split_dir_path = mtedx_path / f"mtedx_{src_lang}" / f"{src_lang}-{src_lang}" / "data" / split
        split_dir_path = mtedx_path / f"{src_lang}-{src_lang}" / "data" / split
        segments_file_path = mtedx_path / f"{src_lang}-{src_lang}" / "data" / split / "txt" / "segments"
        segments_file_lines = read_txt_file(segments_file_path)
        segments_order = list(np.unique(np.array([el.split(' ')[1] for el in segments_file_lines])))
        # print(segments_order)
        # # in_vtt_dir_path = mtedx_path / f"mtedx_{src_lang}" / f"{src_lang}-{src_lang}" / "data" / split / "vtt"
        in_vtt_dir_path = mtedx_path / f"{src_lang}-{src_lang}" / "data" / split / "vtt"

        transcriptions_path = split_dir_path / "txt"
        video_transcriptions = defaultdict(list)
        in_vtt_files = [vtt_filepath for vtt_filepath in sorted(in_vtt_dir_path.glob('**/*'))]
        in_vtt_files = sorted(in_vtt_files, key=lambda x: segments_order.index(x.stem.split('.')[0]))
        # print(in_vtt_files)

        for file in in_vtt_files:
            name = str(file).split('/')[-1].split('.')[0]
            # print('NAME: ', name)
            video_transcriptions[name] = preprocess_vtt(file, duration_threshold)
        
        with open(transcriptions_path / f"transcriptions{duration_threshold}", "w") as transcriptions:
           for key, value in video_transcriptions.items():
               i = 0
               for item in value:
                    transcriptions.write("{}\n".format(f"{item['id']} {key} {item['start']} {item['end']} {item['text']}"))



def preprocess_vtt(vtt_file_path: Path, duration_threshold: int) -> list:
    """
        Find full sentences, which duration is less then duration_threshold seconds.
        Arguments:
            - vtt_file_path - the path to your current vtt file with subtitles
            - duration_threshold - maximum length of a video in seconds
        Return:
            - extracted_lines - a list of dictionaries for each full sentence
    
    """
    with open(vtt_file_path, 'r') as file:
        video_name = str(vtt_file_path).split('/')[-1].split('.')[0]
        lines = file.readlines()
        extracted_lines_ = []
        current_line_ = {'id': '', 'start': None, 'end': None, 'text': ''}
        max_duration = 0
        flag_start = 0
        for line in lines[8:]:
            i = len(extracted_lines_)
            if line.strip():
                if '-->' in line:
                    if flag_start == 0: 
                        start_time = time_to_seconds(line.split('-->')[0].replace(' ', ''))
                        flag_start = 1
                    t = time_to_seconds(line.split('-->')[1].replace('\n', ''))
                    if t - start_time <= duration_threshold: 
                        end_time = t
                        current_line_['id'] = f'{video_name}_{i:04}'
                        current_line_['start'] = start_time
                        current_line_['end'] = end_time
                else:
                    end_line = False
                    if 'Видео' in line: print('Здесь может содержаться фрагмент другого видео, чекнуть: ', video_name)
                    if line.strip()[-1] in ['.', '?', '!']:
                        end_line = True
                        # if video_name == "zY_Edd3AAU0": print(line)
                    line = preprocess_line(line)
                    if len(line) == 0:
                        continue
                    current_line_['text'] += line.upper().strip() + ' '
                    if end_line:
                        if current_line_['start'] is not None and current_line_['end'] is not None:
                            extracted_lines_.append(current_line_)
                        current_line_ = {'id': '', 'start': None, 'end': None, 'text': ''}
                        flag_start = 0

    # print(extracted_lines_)
    return extracted_lines_



def make_transcription_segments(mtedx_path, src_lang, seg_duration):
    """

    """
    for split in SPLITS:
        # split_dir_path = mtedx_path / f"mtedx_{src_lang}" / f"{src_lang}-{src_lang}" / "data" / split
        split_dir_path = mtedx_path / f"{src_lang}-{src_lang}" / "data" / split


        txt_path = split_dir_path / "txt"
        # segment_file = (
        #     mtedx_path / f"mtedx_{src_lang}" / f"{src_lang}-{src_lang}" / "data" / split / "txt" / "segments"
        # )
        segment_file = (
            mtedx_path / f"{src_lang}-{src_lang}" / "data" / split / "txt" / "segments"
        )

        # transcript_file = (
        #     mtedx_path / f"mtedx_{src_lang}" / f"{src_lang}-{src_lang}" / "data" / split / "txt" / "transcriptions"
        # )

        transcript_file = (
            mtedx_path / f"{src_lang}-{src_lang}" / "data" / split / "txt" / f"transcriptions{seg_duration}"
        )
        file = open(txt_path / f'segments_transcriptions{seg_duration}', "w")
        with open(segment_file, 'r') as segments_file, open(transcript_file, 'r') as transcriptions_file:
            segments_lines = segments_file.readlines()
            transcriptions_lines = transcriptions_file.readlines()

            transcriptions_intervals = {}
            for line in transcriptions_lines:
                parts = line.strip().split()
                file_ = parts[1]
                start = float(parts[2])
                end = float(parts[3])
                if file_ in transcriptions_intervals:
                    transcriptions_intervals[file_].append((start, end, " ".join(parts[4:]), parts[0]))
                else:
                    transcriptions_intervals[file_] = [(start, end, " ".join(parts[4:]), parts[0])]

            segment_intervals = {}
            for line in segments_lines:
                parts = line.strip().split()
                file_ = parts[1]
                start = float(parts[2])
                end = float(parts[3])
                if file_ in segment_intervals:
                    segment_intervals[file_].append((start, end))
                else:
                    segment_intervals[file_] = [(start, end)]

            for key in segment_intervals.keys():
                for segm in segment_intervals[key]:
                    start, end = segm
                    i = bisect.bisect_left(transcriptions_intervals[key], (int(start), int(start)))
                    matched_text = None
                    while i < len(transcriptions_intervals[key]):
                        interval_start, interval_end, interval_text, segm_vid = transcriptions_intervals[key][i]
                        if interval_start <= end and interval_end >= start:
                            matched_text = interval_text
                            break
                        i += 1
                    if matched_text:
                        file.write("{}\n".format(
                            f"{segm_vid} {key} {start} {end} {matched_text}"))

    print('end of segm transcr')        

def preprocess_video(mtedx_path, src_lang, dir_out, seg_duration):
    """
        тут надо проходиться по двум файлам и сопоставлять, обрезать видео и создавать соотв файлик с транскрипцией, 
        а также еще общий файл с транскрипцией

        составить новый файл - сегментс + нужная транскрипция
    """
    for split in SPLITS:
        # split_dir_path = mtedx_path / f"mtedx_{src_lang}" / f"{src_lang}-{src_lang}" / "data" / split
        split_dir_path = mtedx_path / f"{src_lang}-{src_lang}" / "data" / split

        video_segments = read_txt_file(split_dir_path / "txt" / "segments")
        out_path = dir_out / split / "video"

        out_txt_path = dir_out / split / "txt"

        out_path.mkdir(parents=True, exist_ok=True)
        out_txt_path.mkdir(parents=True, exist_ok=True)

        try: 
            read_txt_file(f'{dir_out}/{split}/{split}')
            common_file = open(f'{dir_out}/{split}/{split}', "w")
        except: 
            common_file = open(f'{dir_out}/{split}/{split}', "w")

        num_curr_video_segments = len(list(out_path.rglob("*.mp4")))
        if num_curr_video_segments == len(video_segments):
            continue
        if split == "train":
            print(
                f"\nSegmenting `{src_lang}` videos files "
                + "(It takes a few hours to complete)"
            )

        # segment_file = (
        #     mtedx_path / f"mtedx_{src_lang}" / f"{src_lang}-{src_lang}" / "data" / split / "txt" / "segments_transcriptions"
        # )
        segment_file = (
            mtedx_path / f"{src_lang}-{src_lang}" / "data" / split / "txt" / f"segments_transcriptions{seg_duration}"
        )
        video_to_segments = defaultdict(list)

        video_format = "mp4"
        # in_video_dir_path = mtedx_path / f"mtedx_{src_lang}" / f"{src_lang}-{src_lang}" / "data" / split / "video"
        in_video_dir_path = mtedx_path / f"{src_lang}-{src_lang}" / "data" / split / "video"


        for line in read_txt_file(segment_file):
            split_ = line.strip().split()
            seg_id, fid, start_sec, end_sec, text = split_[0], split_[1], split_[2], split_[3], ' '.join(split_[4:])
            # if 
            filepath = out_path / f"{seg_id}.{video_format}"
            if not path.exists(filepath):
                video_to_segments[fid].append(
                    {
                        "id": seg_id,
                        "start": float(start_sec),
                        "end": float(end_sec),
                        "text": text,
                    }
                )
            del line
            gc.collect()


        # video_to_segments = OrderedDict(
        #     sorted(video_to_segments.items(), key=lambda x: len(x[1]))
        # )

        video_format = "mp4"
        # # in_video_dir_path = mtedx_path / f"mtedx_{src_lang}" / f"{src_lang}-{src_lang}" / "data" / split / "video"
        in_video_dir_path = mtedx_path / f"{src_lang}-{src_lang}" / "data" / split / "video"
        print(f'\n==============START PROCESSIND VIDEOS FROM {split} SPLIT==============\n')
        for video_id, video_segments in tqdm(video_to_segments.items()):
            all_segments_are_processed = all(
                (out_path / video_id / f"{seg['id']}.{video_format}").exists()
                for seg in video_segments
            )
            if all_segments_are_processed:
                continue
            in_filepath = in_video_dir_path / f"{video_id}.{video_format}"

            if not in_filepath.exists():
                warnings.warn(
                    f"TED talk `{in_filepath.stem}` hasn't been downloaded..." +
                    " skipping!!" 
                )
                continue

            out_seg_path = out_path / video_id
            out_seg_txt_path = out_txt_path / video_id

            out_seg_path.mkdir(parents=True, exist_ok=True)
            out_seg_txt_path.mkdir(parents=True, exist_ok=True)

            for item in video_segments:
                start = str(datetime.timedelta(seconds=float(item['start'])))
                end = str(datetime.timedelta(seconds=float(item['end'])))
                txt_content = preprocess_line(item['text'])
                file_out = out_seg_path / f"{item['id']}.{video_format}"
                # print(f"-------SEGMENTING {item['id']}-------")
                if not path.exists(file_out):
                    # print(file_out)
                    # continue
                # else:
                    try:
                        (
                        ffmpeg
                        .input(str(in_filepath), ss=start, to=end)
                        .output(str(file_out))
                        .run(quiet=True)
                        )
                        success = True
                    except:
                        success = False
                    
                    if success:
                        with open(f"{out_seg_txt_path / item['id']}.txt", "w") as file:
                            # print(f"-------WRITING TRANSCRIPTION IN {item['id']}-------")
                            file.write("{}".format(f"{item['start']} {item['end']} {txt_content}"))
                        common_file.write("{}\n".format(f"{item['id']} {item['start']} {item['end']} {txt_content}"))
        print(f'-------ALL VIDEOS FROM {split} SPLIT PROCESSED!-------')


def extract_tgz(tgz_filepath, extract_path, out_filename=None):
    if not tgz_filepath.exists():
        raise FileNotFoundError(f"{tgz_filepath} is not found!!")
    tgz_filename = tgz_filepath.name
    tgz_object = tarfile.open(tgz_filepath)
    if not out_filename:
        out_filename = tgz_object.getnames()[0]
    if not (extract_path / out_filename).exists():
        for mem in tqdm(tgz_object.getmembers(), desc=f"\n-------Extracting {tgz_filename}-------\n"):
            out_filepath = extract_path / mem.get_info()["name"]
            if mem.isfile() and not out_filepath.exists():
                tgz_object.extract(mem, path=extract_path)
    tgz_object.close()


def prepare_txt(mtedx_path, src_lang):
    # corpus_path = mtedx_path / f"mtedx_{src_lang}" / f"{src_lang}-{src_lang}" / "data" / f"input_{src_lang}.txt"
    corpus_path = mtedx_path / f"{src_lang}-{src_lang}" / "data" / f"input_{src_lang}"

    corpus_file = open(corpus_path, "w")

    for split in SPLITS:
        # split_dir_path = mtedx_path / f"mtedx_{src_lang}" / f"{src_lang}-{src_lang}" / "data" / split
        split_dir_path = mtedx_path / f"{src_lang}-{src_lang}" / "data" / split

        txt_path = split_dir_path / "txt"
        split_corpus_file = open(txt_path / f"{split}.{src_lang}")
        lines = split_corpus_file.readlines()

        for line in lines:
            line = preprocess_line(line)
            if len(line) == 0:
                continue
            if line.strip() != "" and line.strip() != " ":
                corpus_file.write("{}\n".format(line))
    corpus_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="mTEDx Preprocessing")
    parser.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help="Root directory of preprocessed dataset",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of dataset",
    )
    parser.add_argument(
        "--downloaded-path",
        type=str,
        required=True,
        help="Directory in which mtedx dataset was downloaded",
    )
    parser.add_argument(
        "--seg-duration",
        type=int,
        default=24,
        help="Max duration (second) for each segment, (Default: 24)",
    )
    parser.add_argument(
        "--src-lang",
        required=True,
        choices=["ar", "de", "el", "en", "es", "fr", "it", "pt", "ru"],
        help="The language code for dataset",
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
    # print(seg_duration)
    dataset = args.dataset   
    dst_vid_dir = args.root_dir
    src_lang = args.src_lang
    downloaded_path = args.downloaded_path


    preprocess_vtt_files(Path(downloaded_path), src_lang, seg_duration)
    make_transcription_segments(Path(downloaded_path), src_lang, seg_duration)
    preprocess_video(Path(downloaded_path), src_lang, Path(dst_vid_dir), seg_duration)