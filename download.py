import argparse
from pathlib import Path
import wget
import warnings
import ffmpeg
import os
from os import path
import numpy as np
import tarfile
from tqdm import tqdm
from urllib.error import HTTPError
from pathlib import Path
from pytube import YouTube
from functools import partial
from tqdm.contrib.concurrent import process_map


# SPLITS = ['train', 'test', 'valid']
SPLITS = ['valid']



def download_extract_file_if_not(url, tgz_filepath, download_filename):
    download_path = tgz_filepath.parent
    if not tgz_filepath.exists():
        # download file
        download_file(url, download_path)
    # extract file
    extract_tgz(tgz_filepath, download_path, download_filename)


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


def download_mtedx_data(download_path, src, tgt):
    """Downloads mTEDx data from OpenSLR"""
    tgz_filename = f"mtedx_{src}-{tgt}.tgz" if src != tgt else f"mtedx_{src}.tgz"
    download_extract_file_if_not(
        url=f"https://www.openslr.org/resources/100/{tgz_filename}",
        tgz_filepath=download_path / tgz_filename,
        download_filename=f"{src}-{tgt}"
    )


def download_video_from_youtube(download_path: str, yt_id: str, only_video: bool=True) -> bool:
    """ Downloads a video from YouTube given its id on YouTube
        Arguments:
            - download_path - the path where your video will be stored
            - yt_id - video id on YouTube
            - only_video - download only video without audio (True) or download both (False)

        Return:
            - downloaded - if video was succesfully downloaded (True) or not (False)
    """
    video_out_path = download_path / f"/{yt_id}.mp4"
    # video_out_path = os.path.join(download_path, f'{yt_id}.mp4')
    if path.exists(video_out_path):
        downloaded = True
    else:
        url = f"https://www.youtube.com/watch?v={yt_id}"
        yt = YouTube(url)
        video = yt.streams.filter(only_video=only_video).first()
        video.download(output_path=download_path, filename=yt_id + '.mp4')
        downloaded = True
    return downloaded


def get_video_duration(video_filepath: str) -> float:
    """
        Returns your video duration in ms.
        Arguments:
            - video_filepath - the path to your already downloaded video file
        Return:
            - video duration in ms or warning
    """
    try:
        streams = ffmpeg.probe(video_filepath)["streams"]
        for stream in streams:
            if stream["codec_type"] == "video":
                return float(stream["duration"])
    except:
        warnings.warn(f"Video file: `{video_filepath}` is corrupted... skipping!!")
        return -1


def download_file(url, download_path): # починить, подредачить типы переменных и возврат
    filename = url.rpartition("/")[-1]
    if not path.exists(download_path + '/' + filename):
        try:
            # download file
            print(f"Downloading {filename} from {url}")
            custom_bar = (
                lambda current, total, width=80: wget.bar_adaptive(
                    round(current / 1024 / 1024, 2),
                    round(total / 1024 / 1024, 2),
                    width,
                )
                + " MB"
            )
            wget.download(url, out=os.path.join(download_path, filename), bar=custom_bar)
            print(f"Downloaded {filename} from {url}")

        except Exception as e:
            message = f"Downloading {filename} failed!"
            raise HTTPError(e.url, e.code, message, e.hdrs, e.fp)
    return True


def is_empty(path):
    return any(path.iterdir()) == False

def read_txt_file(txt_filepath):
    with open(txt_filepath) as fin:
        return (line.strip() for line in fin.readlines())
    

def download_mtedx_videos(args):
    try:
        not_found_videos = set(read_txt_file(args["mTedx"] / "not_found_videos.txt"))
    except FileNotFoundError:
        not_found_videos = set()
    
    for split in SPLITS:
        print(args['dataset'])
        download_path = args['mTedx'] / f"{args['dataset']}_{args['src_lang']}"/ f"{args['src_lang']}-{args['src_lang']}" / "video" / split
        download_path.mkdir(parents=True, exist_ok=True)

        if is_empty(download_path): #TODO: better check
            if split == "train":
                print(f"\nDownloading {args['src_lang']} videos from YouTube")
        
            wav_dir_path = (
                args['mTedx'] / f"{args['dataset']}_{args['src_lang']}"/ f"{args['src_lang']}-{args['src_lang']}" / "data" / split / "wav"
            )
            yt_ids = [wav_filepath.stem for wav_filepath in wav_dir_path.glob("*")]
            # wav_files_path = args['mTedx'] / f"{args['dataset']}_{args['src_lang']}"/ f"{args['src_lang']}-{args['src_lang']}" / "data" / split / "wav"
            # print(download_path)
            downloading_status = process_map(
                partial(download_video_from_youtube, download_path),
                yt_ids,
                max_workers=os.cpu_count(),
                desc=f"Downloading {args['src_lang']}/{split} Videos",
                chunksize=1,
            )
            assert len(yt_ids) == len(downloading_status)
            for yt_id, downloaded in zip(yt_ids, downloading_status):
                if not downloaded:
                    not_found_videos.add(yt_id)
    with open(args['mTedx'] / "not_found_videos.txt", "w") as fout:
        fout.writelines([f"{id_}\n" for id_ in not_found_videos])



def prepare_mtedx(args):

    # download mtedx dataset if needed
    if args["download"]==1:
        download_mtedx_data(args["mTedx"], args["src_lang"], args["src_lang"])

    # download mtedx videos from youtube
    download_mtedx_videos(args)

    


def main(args):
    dirs = ["mTedx"]
    for dirname in dirs:
        args[dirname] = args["root_path"] / dirname
        args[dirname].mkdir(parents=True, exist_ok=True)
    prepare_mtedx(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["mtedx"],
        help="The dataset.",
    )
    parser.add_argument(
        "--download",
        default=False,
        # required=True,
        type=int,
        choices=[1, 0],
        help="Download the dataset or not.",
    )
    parser.add_argument(
        "--root-path",
        required=True,
        type=Path,
        help="Relative/Absolute path where MuAViC dataset will be downloaded.",
    )
    parser.add_argument(
        "--src-lang",
        required=True,
        choices=["ar", "de", "el", "en", "es", "fr", "it", "pt", "ru"],
        help="The language code for dataset",
    )
    parser.add_argument(
        "--num-workers",
        default=os.cpu_count(),
        help="Max number of workers to be used in parallel.",
    )
    args = vars(parser.parse_args())
    main(args)

