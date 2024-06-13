import yaml
from collections import defaultdict

# Загрузка данных из файла YAML
with open('/home/sadevans/space/personal/preprocess_mtedx/parse/unique_words_and_speakers.yaml', 'r', encoding='utf-8') as file:
    data = yaml.safe_load(file)

# print(data.keys())
# Создание словаря для хранения информации о словах и спикерах
word_speakers = defaultdict(set)

# # Обработка данных
for word, values in data.items():
    print(word, values)
    # break
    for item in values:
        speaker_id = item['speaker_id']
        if speaker_id not in word_speakers[word]: word_speakers[word].add(speaker_id)


sorted_words_by_speakers = sorted(word_speakers.items(), key=lambda x: len(x[1]), reverse=True)

with open("word_num_speaker.yaml", "w", encoding='utf-8') as file:
    for word, speakers in sorted_words_by_speakers:
        data = {word: len(speakers)}
        yaml.safe_dump(data, file, allow_unicode=True)

