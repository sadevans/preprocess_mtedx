#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# https://github.com/pytorch/fairseq/blob/master/LICENSE
import sys
import io
import sentencepiece as spm



if __name__ == "__main__":
    model_writer = io.BytesIO()

    spm.SentencePieceTrainer.Train(" ".join(sys.argv[1:5]), model_writer=model_writer, input_sentence_size=-1, \
                                   character_coverage=1.0, bos_id=5, pad_id=2, eos_id=3, unk_id=4)
    
    
    # model_writer = io.BytesIO()
    # spm.SentencePieceTrainer.Train(
    #     sentence_iterator=iter(sys.argv[1].split("=")[-1]),
    #     model_writer=model_writer,
    #     vocab_size=int(sys.argv[2].split("=")[-1]),
    #     model_type="unigram",
    #     model_prefix = sys.argv[4].split('=')[-1],
    #     input_sentence_size=-1,
    #     character_coverage=1.0,
    #     bos_id=0,
    #     pad_id=1,
    #     eos_id=2,
    #     unk_id=3,
    # )
