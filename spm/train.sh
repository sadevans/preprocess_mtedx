nbpe=1023
bpemode=unigram
mkdir -p ${bpemode}
dict=${bpemode}_ru/${bpemode}${nbpe}_units.txt
bpemodel=${bpemode}_ru/${bpemode}${nbpe}_ru
echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
python spm_train.py --input=input_ru --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
python spm_encode.py --model=${bpemodel}.model --output_format=piece < input_ru | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}