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


# SPLITS = ['train', 'test', 'valid']
SPLITS = ['valid']


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


def read_txt_file(txt_filepath:Path):
    """
        Read text files
        Arguments:
            - txt_filepath- the path to your txt file
        Return:
            all lines from your txt file
    """
    with open(txt_filepath) as fin:
        return (line.strip() for line in fin.readlines())


def download_mtedx_data(download_path, src, tgt):
    """Downloads mTEDx data from OpenSLR"""
    tgz_filename = f"mtedx_{src}-{tgt}.tgz" if src != tgt else f"mtedx_{src}.tgz"
    download_extract_file_if_not(
        url=f"https://www.openslr.org/resources/100/{tgz_filename}",
        tgz_filepath=download_path / tgz_filename,
        download_filename=f"{src}-{tgt}"
    )



def preprocess_vtt_files(mtedx_path: Path, src_lang: str, duration_threshold: int) -> None:
    """
        Make transcriptions file for video segments for all videos in all splits.
        Arguments:
            - mtedx_path - the path to mtedx dataset
            - src_lang - source language
            - duration_threshold - maximum length of a video segment in seconds
        Return:
            None
    """
    for split in SPLITS:
        split_dir_path = mtedx_path / f"mtedx_{src_lang}" / f"{src_lang}-{src_lang}" / "data" / split
        in_vtt_dir_path = mtedx_path / f"mtedx_{src_lang}" / f"{src_lang}-{src_lang}" / "data" / split / "vtt"

        transcriptions_path = split_dir_path / "txt"
        video_transcriptions = defaultdict(list)
        for file in in_vtt_dir_path.iterdir():
            name = str(file).split('/')[-1].split('.')[0]
            video_transcriptions[name] = preprocess_vtt(file, duration_threshold)

        video_transcriptions = OrderedDict(
            sorted(video_transcriptions.items(), key=lambda x: len(x[1]))
        )
        with open(transcriptions_path / "transcriptions.txt", "w") as transcriptions:
           for key, value in video_transcriptions.items():
            #    print('LEN:', len(value))
               i = 0
               for item in value:
                    if i != len(value) - 1:
                        transcriptions.write(f"{item['id']} {key} {item['start']} {item['end']} {item['text']}\n")
                    else:
                        transcriptions.write(f"{item['id']} {key} {item['start']} {item['end']} {item['text']}")
                    i += 1



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
        # print(vtt_file_path)

        video_name = str(vtt_file_path).split('/')[-1].split('.')[0]
        # print(video_name)
        lines = file.readlines()
        # extracted_lines = []
        extracted_lines_ = []
        # current_line = {'time': [], 'text': ''}
        current_line_ = {'id': '', 'start': None, 'end': None, 'text': ''}
        # print('CURRENT LINE LEN', len(current_line_['id']))

        max_duration = 0
        flag_start = 0

        for line in lines[8:]:
            # if 
            i = len(extracted_lines_)
            if not any(char in line for char in '()[]'):
                if line.strip():  # проверка на пустую строку
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

                            # current_line['time'] = [start_time, end_time]
                        if end_time - start_time > max_duration:
                            max_duration = end_time - start_time
                    else:
                        # current_line['text'] += line.strip() + ' '
                        current_line_['text'] += line.strip() + ' '
                        if line.strip()[-1] in ['.', '?', '!']:
                            # extracted_lines.append(current_line['text'].strip())
                            # if len(current_line['time']) == 2: 
                            if current_line_['start'] is not None and current_line_['end'] is not None:
                            
                                # extracted_lines.append(current_line)
                                extracted_lines_.append(current_line_)

                            # current_line = {'time': [], 'text': ''}
                            current_line_ = {'id': '', 'start': None, 'end': None, 'text': ''}
                            flag_start = 0

    return extracted_lines_



def make_transcription_segments(mtedx_path, src_lang):
    """
    составить новый файл - сегментс + нужная транскрипция

    не работает послк случая 
    типа смотри в первом файле может быть так:
    178 180
    180 183

    а во втором вот так:
    178 183
    """

    for split in SPLITS:
        split_dir_path = mtedx_path / f"mtedx_{src_lang}" / f"{src_lang}-{src_lang}" / "data" / split

        txt_path = split_dir_path / "txt"
        # segments_txt = txt_path / "segments.txt"
        # transcript_txt = txt_path / "transcriptions.txt"

        segment_file = (
            mtedx_path / f"mtedx_{src_lang}" / f"{src_lang}-{src_lang}" / "data" / split / "txt" / "segments.txt"
        )

        transcript_file = (
            mtedx_path / f"mtedx_{src_lang}" / f"{src_lang}-{src_lang}" / "data" / split / "txt" / "transcriptions.txt"
        )

        video_transcript_segments = defaultdict(list)

        read_transcript_file = read_txt_file(transcript_file)
        read_segment_file = read_txt_file(segment_file)
        i = 0
        # with open(txt_path / 'segments_transcriptions.txt', "w") as file:
        #     for line_segm, line_transcript in zip(read_segment_file, read_transcript_file):
        #         # print(' '.join(line_transcript.strip().split()[4:]))
        #         split_ = line_transcript.strip().split()
        #         seg_id, fid, start_sec, end_sec = line_segm.strip().split()
        #         seg_id_tr, fid_tr, start_sec_tr, end_sec_tr, text = split_[0], split_[1], split_[2], split_[3], ' '.join(split_[4:])
        #         if fid == fid_tr and (int(float(start_sec)) == int(float(start_sec_tr))) and (int(float(end_sec)) == int(float(end_sec_tr))):
        #             file.write(f"{seg_id} {fid} {start_sec} {end_sec} {text}\n")

        #         elif fid == fid_tr and (int(float(start_sec)) == int(float(start_sec_tr))) and (int(float(end_sec)) != int(float(end_sec_tr))):
        #             file.write(f"{seg_id} {fid} {start_sec} {end_sec_tr} {text}\n")
        #         else:
        #             continue
        with open(txt_path / 'segments_transcriptions.txt', "w") as file:
            with open(segment_file, 'r') as file_seg, open(transcript_file, 'r') as file_tr:
                for line_seg in file_seg:
                    line_tr = file_tr.readline()
                    seg_id, fid, start_sec, end_sec = line_seg.strip().split()
                    split_ = line_tr.strip().split()
                    if len(split_) > 0:
                        seg_id_tr, fid_tr, start_sec_tr, end_sec_tr, text = split_[0], split_[1], split_[2], split_[3], ' '.join(split_[4:])

                        # line_tr = next(file_tr, line_tr)
                        # split_ = line_tr.strip().split()
                        # print(split_)
                        if len(split_) > 0:
                            
                            seg_id_tr, fid_tr, start_sec_tr, end_sec_tr, text = split_[0], split_[1], split_[2], split_[3], ' '.join(split_[4:])
                            
                            if fid == fid_tr and int(float(start_sec)) == int(float(start_sec_tr)) and (int(float(end_sec)) == int(float(end_sec_tr))):
                                file.write(f"{seg_id} {fid} {start_sec} {end_sec} {text}\n")
                            elif fid == fid_tr and (int(float(start_sec)) == int(float(start_sec_tr))) and (int(float(end_sec)) != int(float(end_sec_tr))):
                                file.write(f"{seg_id_tr} {fid} {start_sec} {end_sec_tr} {text}\n")
                            else:
                                continue


def preprocess_video(mtedx_path, src_lang, dir_out):
    """
        тут надо проходиться по двум файлам и сопоставлять, обрезать видео и создавать соотв файлик с транскрипцией, 
        а также еще общий файл с транскрипцией

        составить новый файл - сегментс + нужная транскрипция
    """
    # mean_face_metadata = load_meanface_metadata(metadata_path)
    for split in SPLITS:
        # print(split)
        split_dir_path = mtedx_path / f"mtedx_{src_lang}" / f"{src_lang}-{src_lang}" / "data" / split
        video_segments = list(read_txt_file(split_dir_path / "txt" / "segments.txt"))
        # out_path = dir_out / src_lang / split / "video"
        out_path = dir_out / split / "video"

        out_txt_path = dir_out / split / "txt"

        out_path.mkdir(parents=True, exist_ok=True)
        out_txt_path.mkdir(parents=True, exist_ok=True)

        try: 
            read_txt_file(f'{dir_out}/{split}/{split}.txt')
            common_file = open(f'{dir_out}/{split}/{split}.txt', "a")
        except: 
            common_file = open(f'{dir_out}/{split}/{split}.txt', "a")

        num_curr_video_segments = len(list(out_path.rglob("*.mp4")))
        if num_curr_video_segments == len(video_segments):
            continue # skip if all video segments are already processed
        if split == "train":
            print(
                f"\nSegmenting `{src_lang}` videos files "
                + "(It takes a few hours to complete)"
            )

        segment_file = (
            mtedx_path / f"mtedx_{src_lang}" / f"{src_lang}-{src_lang}" / "data" / split / "txt" / "segments_transcriptions.txt"
        )
        video_to_segments = defaultdict(list)

        video_format = "mp4"
        in_video_dir_path = mtedx_path / f"mtedx_{src_lang}" / f"{src_lang}-{src_lang}" / "data" / split / "video"

        for line in read_txt_file(segment_file):
            split_ = line.strip().split()
            seg_id, fid, start_sec, end_sec, text = split_[0], split_[1], split_[2], split_[3], ' '.join(split_[4:])
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


        video_to_segments = OrderedDict(
            sorted(video_to_segments.items(), key=lambda x: len(x[1]))
        )

        video_format = "mp4"
        in_video_dir_path = mtedx_path / f"mtedx_{src_lang}" / f"{src_lang}-{src_lang}" / "data" / split / "video"

        for video_id, video_segments in tqdm(video_to_segments.items()):
            # check if video has been already processed:
            all_segments_are_processed = all(
                (out_path / video_id / f"{seg['id']}.{video_format}").exists()
                for seg in video_segments
            )
            if all_segments_are_processed:
                continue
            # prepare to process video file
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

            print(f'START PROCESSIND VIDEOS FROM {split} SPLIT....')
            for i, item in enumerate(video_segments):
                start = str(datetime.timedelta(seconds=float(item['start'])))
                end = str(datetime.timedelta(seconds=float(item['end'])))
                txt_content = item['text']
                file_out = out_seg_path / f"{item['id']}.{video_format}"
                (
                    ffmpeg
                    .input(str(in_filepath), ss=start, to=end)
                    .output(str(file_out))
                    .run(quiet=True)
                )
                with open(f"{out_seg_txt_path / item['id']}.txt", "w") as file:
                    file.write(f"{item['start']} {item['end']} {txt_content}")

                if i != len(video_segments) - 1: common_file.write(f"{item['id']} {item['start']} {item['end']} {txt_content}\n")
                else: common_file.write(f"{item['id']} {item['start']} {item['end']} {txt_content}")
            print(f'ALL VIDEOS FROM {split} SPLIT PROCESSED !')



def extract_tgz(tgz_filepath, extract_path, out_filename=None):
    if not tgz_filepath.exists():
        raise FileNotFoundError(f"{tgz_filepath} is not found!!")
    tgz_filename = tgz_filepath.name
    tgz_object = tarfile.open(tgz_filepath)
    if not out_filename:
        out_filename = tgz_object.getnames()[0]
    # check if file is already extracted
    if not (extract_path / out_filename).exists():
        for mem in tqdm(tgz_object.getmembers(), desc=f"Extracting {tgz_filename}"):
            out_filepath = extract_path / mem.get_info()["name"]
            if mem.isfile() and not out_filepath.exists():
                tgz_object.extract(mem, path=extract_path)
    tgz_object.close()


if __name__ == '__main__':
    # Argument Parsing
    parser = argparse.ArgumentParser(description="mTEDx Preprocessing")
    parser.add_argument(
        "--detector",
        type=str,
        default="retinaface",
        help="Type of face detector. (Default: retinaface)",
    )
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
    dataset = args.dataset   
    dst_vid_dir = os.path.join(
        args.root_dir, dataset
    )
    src_lang = args.src_lang
    downloaded_path = args.downloaded_path


    preprocess_vtt_files(Path(downloaded_path), src_lang, seg_duration)
    make_transcription_segments(Path(downloaded_path), src_lang)
    preprocess_video(Path(downloaded_path), src_lang, Path(dst_vid_dir))
# 
