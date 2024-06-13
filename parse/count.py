import yaml
from collections import defaultdict

# Загрузка данных из файла YAML
with open('/media/sadevans/T7/PERSONAL/Diplom/datsets/Vmeste/ru-ru/data/all/txt/all.yaml', 'r', encoding='utf-8') as file:
    data = yaml.safe_load(file)

# Создание словаря для хранения информации о словах для каждого спикера
word_speakers = defaultdict(list)

# Обработка данных
for item in data:
    speaker_id = item['speaker_id']
    word = item['word']
    start = item['start']
    end = item['end']

    # Добавление спикера и временных меток для каждого слова
    word_speakers[word].append({'speaker_id': speaker_id, 'start': start, 'end': end})

# Создание словаря для уникальных слов и списков спикеров
unique_words = {}
for word, speakers in word_speakers.items():
    unique_words[word] = speakers

# Запись результатов в отдельный файл YAML
with open('unique_words_and_speakers.yaml', 'w', encoding='utf-8') as file:
    yaml.safe_dump(unique_words, file, allow_unicode=True)
