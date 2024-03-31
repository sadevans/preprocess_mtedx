import os
import re
import argparse
from pathlib import Path

SPLITS = ['train', 'test', 'valid']


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



def prepare_txt(mtedx_path, src_lang):
    corpus_path = mtedx_path / f"mtedx_{src_lang}" / f"{src_lang}-{src_lang}" / "data" / f"input_{src_lang}"
    # corpus_path = f"input_{src_lang}.txt"

    corpus_file = open(corpus_path, "w")

    for split in SPLITS:
        split_dir_path = mtedx_path / f"mtedx_{src_lang}" / f"{src_lang}-{src_lang}" / "data" / split
        txt_path = split_dir_path / "txt"
        split_corpus_file = open(txt_path / f"{split}.{src_lang}")
        lines = split_corpus_file.readlines()

        for i, line in enumerate(lines):
            # line = line.upper()
            # line = re.sub(r'^[^:]+:\s*', '', line)
            # line = re.sub(r'\([^)]*\)', '', line)
            # line = line.replace('Ё', 'Е')
            # line = line.replace('ё', 'е')
            # line = re.sub(r'[^0-9а-яА-Я- ]', '', line)
            # line = line.replace('\t', '')
            # line = re.sub(r'\s{2,}', ' ', line)
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
        "--src-lang",
        required=True,
        choices=["ar", "de", "el", "en", "es", "fr", "it", "pt", "ru"],
        help="The language code for dataset",
    )
    args = parser.parse_args()

    dataset = args.dataset   
    dst_vid_dir = os.path.join(
        args.root_dir, dataset
    )
    src_lang = args.src_lang
    downloaded_path = args.downloaded_path

    prepare_txt(Path(downloaded_path), src_lang)