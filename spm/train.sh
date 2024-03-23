nbpe=5000
bpemode=unigram
mkdir -p ${bpemode}
dict=${bpemode}_ru/${bpemode}${nbpe}_units_ru.txt
bpemodel=${bpemode}_ru/${bpemode}${nbpe}
echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
python spm_train.py --input=input_ru.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
python spm_encode.py --model=${bpemodel}.model --output_format=piece < input_ru.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
