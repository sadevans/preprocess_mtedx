import os
import yaml
import json
import torch
import torchaudio
import torchvision
from tqdm import tqdm
from joblib import Parallel, delayed
import re



def preprocess_text(line):
    # line = line.upper()
    line = line.lower()
    line = re.sub(r'\([^(){}\[\]]*\)', '', line)
    line = line.replace('Ё', 'Е')
    line = line.replace('ё', 'е')
    line = re.sub(r'[^0-9а-яА-Я- ]', '', line)
    line = line.replace('\t', '')
    line = re.sub(r'\s{2,}', ' ', line)

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
    # parse YAML file containing video (audio) info
    with open( os.path.join(base_dir, f"{group}.yaml"), "r") as fin:
        video_samples = yaml.load(fin, Loader=yaml.Loader)
    # parse text file containing text info
    with open( os.path.join(base_dir, f"{group}.{lang}"), "r") as fin:
        text_samples = fin.readlines()
    # sanity check
    assert len(text_samples) == len(video_samples), \
        f"Data mismatch with language: {lang}, group: {group}"
    
    print(video_samples)
    return video_samples, text_samples


def split_and_save_video(input_path, output_path, start_time, end_time,
        input_fps, out_fps=25):
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
    video_frames, _, metadata = torchvision.io.read_video(input_path, start_pts=start_time, end_pts=end_time)

    print(len(video_frames))
    print(int(end_time - start_time)*input_fps)
    torchvision.io.write_video(output_path, video_frames, fps=out_fps, video_codec='libx264')


def process_video_text_sample(i, video, text, data_folder, save_folder,
        lang, group):
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

    if len(text) == 0:
        return None
    
    video_input_filepath = (
        f"{data_folder}/{lang}-{lang}/data/{group}/video/{video['speaker_id']}.mp4"
    )
    video_segment_filename = video["speaker_id"]+f"_{i:04d}"
    video_output_filepath = (
        f"{save_folder}/{lang}/{group}/{video_segment_filename}.mp4"
    )
    # save audio file
    _, _ , info = torchvision.io.read_video(video_input_filepath)
    split_and_save_video(
        video_input_filepath,
        video_output_filepath,
        video["offset"],
        video["offset"]+video["duration"],
        info.video_fps,
        OUT_FPS
    )
    # ruturn a line
    return f"{video_segment_filename} {video['offset']} {video['offset']+video['duration']} {text}"



def preprocess(data_folder, save_folder, lang, group):
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
    os.makedirs(save_folder, exist_ok=True)
    lang_folder = os.path.join(save_folder, lang)
    os.makedirs(lang_folder, exist_ok=True)
    os.makedirs(os.path.join(lang_folder, group), exist_ok=True)
    
    # Setting path for the group file
    group_file = os.path.join(lang_folder, f"{group}.txt")

    # skip if the file already exists
    if os.path.exists(group_file):
        print(f"{group_file} already exists. Skipping!!")
        return
    
    print(f"Creating group file in {group_file} for {lang}, group: {group}.txt")
    video_samples, text_samples = load_video_text_data(data_folder, lang, group)
    # combine text & video information
    result = Parallel(n_jobs=2, backend="threading")(
        delayed(process_video_text_sample)
        (i, video, text, data_folder, save_folder, lang, group) \
        for i, (video, text,) in tqdm(
            enumerate(zip(video_samples, text_samples)),
            desc=f"Processing {lang}, {group}",
            total = len(video_samples)
        )
    )
    
    # write line into group file   
    with open(group_file, 'w', encoding='utf8') as fout:
        for line in result:
            if line is not None:
                fout.write("{}\n".format(line))
    print(f"{group_file} successfully created!")


def main(args):
    """
    Main function that iterates over all languages and groups to process
    the audio data.
    """
    for lang in args["langs"]:
        for group in args["groups"]:
            preprocess(args["in"], args["out"], lang, group)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', type=str, required=True,
                help='The absolute/relative path where the downloaded mTEDx'+ \
                    'data are located.')
    parser.add_argument('--out', type=str, required=True,
                help='The absolute/relative path where the processed mTEDx'+ \
                    'data will be located.')
    parser.add_argument('--langs', nargs='+', required=True,
                help='List of language codes separated by space, eg "de fr"')
    parser.add_argument('--groups', nargs='+', default="test valid train",
                help='List of groups separated by space, e.g. "valid train".')
    parser.add_argument('--fps', type=int, default=25,
                help='The sample rate of the output video files.')
    # parse arguments
    args = vars(parser.parse_args())

    # Global variable
    OUT_FPS = args["fps"]

    # process the data
    main(args)