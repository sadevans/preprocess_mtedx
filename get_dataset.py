import argparse
from pathlib import Path

from mtedx_utils import *


def prepare_mtedx(args):
    if args["download"]==1:
        download_mtedx_data(args["mTedx"], args["src_lang"], args["src_lang"])
    else:
        print('don\'t download')
        print(args['mTedx'])

    # Download mTedx videos from Youtube
    download_mtedx_lang_videos(args["mTedx"], args["src_lang"])

    # Preprocess mTedx downloaded videos
    preprocess_mtedx_video(
        args["mTedx"], args["metadata"], args["src_lang"], args["muavic"]
    )

    # Make avsr manifests
    prepare_mtedx_avsr_manifests(args["mTedx"], args["src_lang"], args["muavic"])


def main(args):
    # created needed directories
    dirs = ["mTedx", "LRS3", 'muavic', 'metadata', 'mt_trans']
    for dirname in dirs:
        args[dirname] = args["root_path"] / dirname
        args[dirname].mkdir(parents=True, exist_ok=True)

    # Prepare mTEDx data
    prepare_mtedx(args)

    # clear out un-needed directories
    shutil.rmtree(args["mt_trans"])
    shutil.rmtree(args["metadata"])

    print(f"Creating {args['dataset']}-{args['src_lang']} is completed!! \u2705")


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        
        required=True,
        choices=["mtedx", "lrs3"],
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
        help="The language code for the source language in MuAViC.",
    )
    parser.add_argument(
        "--num-workers",
        default=os.cpu_count(),
        help="Max number of workers to be used in parallel.",
    )
    args = vars(parser.parse_args())
    print(args)
    main(args)

    