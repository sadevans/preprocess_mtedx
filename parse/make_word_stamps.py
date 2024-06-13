from googleapiclient.discovery import build
import argparse
from sklearn.model_selection import train_test_split
import assemblyai as aai
import os
from os import path
from pytube import YouTube
import yaml
import re

with open('./configs/parse.yaml', 'r') as file:
    config = yaml.safe_load(file)
YT_API_KEY = config['yt_api_key']
AAI_API_KEY = config['aai_api_key']
aai.settings.api_key = AAI_API_KEY
# SUBSETS = ['valid', 'test', 'train']
# SUBSETS = ['valid']
# SUBSETS = ['test']
SUBSETS = ['train']

def download_video_from_youtube(download_path: str, yt_id: str, only_video: bool=True) -> bool:
    """ Downloads a video from YouTube given its id on YouTube
        Arguments:
            - download_path - the path where your video will be stored
            - yt_id - video id on YouTube
            - only_video - download only video without audio (True) or download both (False)

        Return:
            - downloaded - if video was succesfully downloaded (True) or not (False)
    """
    # video_out_path = download_path / f"/{yt_id}.mp4"

    video_out_path = path.join(download_path, f'{yt_id}.mp4')
    audio_out_path = os.path.join(download_path.replace('video', 'wav'), yt_id + '.flac')

    if path.exists(audio_out_path):
        downloaded_aud = True
        try:
            audio = yt.streams.filter(only_audio=True,file_extension='webm').first()
            audio.download(output_path=download_path.replace('video', 'wav'), filename=yt_id + '.flac')
            downloaded_aud = True
        except:
            downloaded_aud=False

    if downloaded_aud:
        if path.exists(video_out_path):
            downloaded = True
        else:
            url = f"https://www.youtube.com/watch?v={yt_id}"
            # print(yt_id)
            try:
                yt = YouTube(url)
                video = yt.streams.filter(only_video=True,res='1080p', file_extension='mp4').first()
                video.download(output_path=download_path, filename=yt_id + '.mp4')

                # audio = yt.streams.filter(only_audio=True,file_extension='webm').first()
                # audio.download(output_path=download_path.replace('video', 'wav'), filename=yt_id + '.flac')
                downloaded = True
            except:
                downloaded=False
    else: downloaded=False
    return downloaded & downloaded_aud



def get_channel_id(youtube, channel_name):
    search_response = youtube.search().list(
        q=channel_name,
        part='snippet',
        type='channel',
        maxResults=1
    ).execute()

    # Extract the channel ID from the search results
    channel_id = search_response['items'][0]['snippet']['channelId']

    return channel_id

def get_channel_videos(src_lang="ru", count_vids=50, channel_name=None):
    youtube = build('youtube', 'v3', developerKey=YT_API_KEY)
    channel_id = get_channel_id(youtube, channel_name)
    uploads_playlist_id = youtube.channels().list(id=channel_id, part='contentDetails').execute()['items'][0]['contentDetails']['relatedPlaylists']['uploads']
    playlist_items = []
    next_page_token = None

    while True:
        playlist_request = youtube.playlistItems().list(
            playlistId=uploads_playlist_id,
            part='snippet',
            maxResults=count_vids,
            pageToken=next_page_token
        )
        playlist_response = playlist_request.execute()
        playlist_items += playlist_response['items']
        next_page_token = playlist_response.get('nextPageToken')

        if not next_page_token:
            break
    video_ids = [item['snippet']['resourceId']['videoId'] for item in playlist_items]

    # video_urls = [ if youtube.captions().list(part='snippet',videoId=video_id).execute()['items']['language'] == src_lang 'https://www.youtube.com/watch?v=' + video_id for video_id in video_ids]
        
    # video_urls = []
    # for video_id in video_ids:
    #     print(youtube.captions().list(part='snippet',videoId=video_id).execute()['items'])
    #     if youtube.captions().list(part='snippet',videoId=video_id).execute()['items'][0]['snippet']['language'] == src_lang:
    #         video_urls.append('https://www.youtube.com/watch?v=' + video_id)


    video_urls = ['https://www.youtube.com/watch?v=' + video_id for video_id in video_ids]

    return video_urls


def search_videos(src_lang="ru", count_vids=50, query=None):
    youtube = build('youtube', 'v3', developerKey=YT_API_KEY)
    search_response = youtube.search().list(
        q=query,
        part='snippet',
        type='video',
        maxResults=count_vids
    ).execute()

    # Extract video IDs from search results
    video_ids = [item['id']['videoId'] for item in search_response['items']]

    # Generate video URLs
    video_urls = ['https://www.youtube.com/watch?v=' + video_id for video_id in video_ids]
    # video_urls = []
    # for video_id in video_ids:
    #     if youtube.captions().list(part='snippet',videoId=video_id).execute()['items']['language'] == src_lang:
    #         video_urls.append('https://www.youtube.com/watch?v=' + video_id)

    return video_urls


def download_audio_video(root_dir, dataset, src_lang):
    for sub in SUBSETS:
        sub_file_path = os.path.join(root_dir, dataset,src_lang+"-"+src_lang, "data", sub, "txt", "videos")
        not_downloaded_file_path = os.path.join(root_dir, dataset,src_lang+"-"+src_lang, "data", sub, "txt", "not_downloaded_videos")
        file = open(not_downloaded_file_path, "a")
        try:
            sub_file = open(sub_file_path, "r").readlines()
        except:
            print(f"There is no {sub_file_path} file !")
            continue

        # print(sub_file)
        download_path = os.path.join(root_dir, dataset, src_lang+"-"+src_lang, "data", sub, "video")
        vtt_path = os.path.join(root_dir, dataset, src_lang+"-"+src_lang, "data", sub, "vtt")
        downloading_status = []
        for url in sub_file:
            video_id = url.replace('\n', '').split("=")[-1]
            # print(video_id)
            if not path.exists(os.path.join(download_path, video_id + '.mp4')):
                download = download_video_from_youtube(download_path, video_id)
                if not download:
                    file.write("{}\n".format(f"{url}"))

def milliseconds_to_seconds(milliseconds):
    seconds = milliseconds / 1000
    return seconds


def get_vtt(root_dir, dataset, src_lang):
    aai.settings.api_key = AAI_API_KEY
    config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.best, language_code=src_lang)
    transcriber = aai.Transcriber(config=config)

    # sentences_file_path = os.path.join(root_dir, dataset, src_lang+"-"+src_lang, "doc", f"input.{src_lang}")
    # if not path.exists(sentences_file_path):
    #     sent_file = open(sentences_file_path, "w")
    #     flag_open = True
    # else: flag_open = False
    # for sub in SUBSETS:
    vtt_path = os.path.join(root_dir, dataset, src_lang+"-"+src_lang, "data", "all", "vtt")
    wav_path = os.path.join(root_dir, dataset, src_lang+"-"+src_lang, "data", "all", "wav")
    yaml_sub_path = os.path.join(root_dir, dataset, src_lang+"-"+src_lang, "data", "all", "txt", "all" + '.yaml')
    segm_sub_path = os.path.join(root_dir, dataset, src_lang+"-"+src_lang, "data", "all", "txt", "segments")

    if not path.exists(yaml_sub_path): sub_yaml = open(yaml_sub_path, "w", encoding='utf-8')
    else: sub_yaml = open(yaml_sub_path, "a")

    if not path.exists(segm_sub_path): sub_segm = open(segm_sub_path, "w")
    else: sub_segm = open(segm_sub_path, "a", encoding='utf-8')

    files = os.listdir(wav_path)
    ind = files.index('CXZlKUABhUI.flac')
    # print(files, ind)
    for file in files[113:]:
        print(file)
        name = file.split('/')[-1].replace('.flac', '')
        if not path.exists(os.path.join(vtt_path, name + '.vtt')):

            # print(name)
            transcript = None
            try:
                transcript = transcriber.transcribe(os.path.join(wav_path, file))
            except:
                transcript = None

                continue
            if transcript is not None:
                for word in transcript.words:
                    
                    data = {"start": milliseconds_to_seconds(word.start), "end": milliseconds_to_seconds(word.end),  "speaker_id": name, "wav": name + '.flac', "word": re.sub(r'[^а-я]', '', word.text.lower())}
                    yaml.dump([data], sub_yaml, default_flow_style=None,  allow_unicode=True, encoding='utf-8')




def parse_urls(root_dir, dataset, count, src_lang, channel=None, query=None):
    if channel: 
        # channel_name = channel
        video_urls = get_channel_videos(src_lang=src_lang, count_vids=count,channel_name=channel)
    if query: 
        # query = query
        video_urls = search_videos(src_lang=src_lang, count_vids=count, query=query)

    train_urls, test_valid_urls = train_test_split(video_urls, test_size=0.2, random_state=42)
    valid_urls, test_urls = train_test_split(test_valid_urls, test_size=0.5, random_state=42)

    print("TOTAL LEN: {}, TRAIN SPLIT LEN : {} | VALID SPLIT LEN: {} | TEST SPLIT LEN: {}".format(len(video_urls), len(train_urls), len(valid_urls), len(test_urls)))

    for sub in SUBSETS:
        os.makedirs(os.path.join(root_dir, dataset, src_lang+"-"+src_lang, "data", sub, "txt"), exist_ok=True)
        os.makedirs(os.path.join(root_dir, dataset, src_lang+"-"+src_lang, "data", sub, "vtt"), exist_ok=True)
        os.makedirs(os.path.join(root_dir, dataset, src_lang+"-"+src_lang, "data", sub, "video"), exist_ok=True)
        os.makedirs(os.path.join(root_dir, dataset, src_lang+"-"+src_lang, "data", sub, "wav"), exist_ok=True)
        sub_file_path = os.path.join(root_dir, dataset, src_lang+"-"+src_lang, "data", sub, "txt", "videos")
        sub_file = open(sub_file_path, "w")
        if sub == "train":
            for url in train_urls:
                sub_file.write("{}\n".format(f"{url}"))

        elif sub == "valid":
            for url in valid_urls:
                sub_file.write("{}\n".format(f"{url}"))
            
        elif sub == "test":
            for url in test_urls:
                sub_file.write("{}\n".format(f"{url}"))


def main(args):
    # if args.if_parse and args.channel:
    #     parse_urls(args.root_dir, args.dataset, args.count, args.src_lang, channel=args.channel)
    # elif args.if_parse and args.query:
    #     parse_urls(args.root_dir, args.dataset, args.count, args.src_lang, query=args.query)
    # elif not args.if_parse: # ???????????

    # os.makedirs(os.path.join(args.root_dir, args.dataset, args.src_lang+"-"+args.src_lang, "doc"), exist_ok=True)
    # download_audio_video(args.root_dir, args.dataset, args.src_lang)
    get_vtt(args.root_dir, args.dataset, args.src_lang)


    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parsing videos from youtube")
    parser.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help="Root directory of where videos will be downloaded",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of dataset",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=False,
        help="Query for searching videos",
    )
    parser.add_argument(
        "--channel",
        type=str,
        required=False,
        help="Channel for searching videos in",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        required=False,
        help="Videos count",
    )
    parser.add_argument(
        "--src-lang",
        type=str,
        required=True,
        help="Source language",
    )
    parser.add_argument(
        "--if-parse",
        type=bool,
        default=False,
        required=False,
        help="Flag if parsing of video urls is needed",
    )
    args = parser.parse_args()
    
    
    main(args)