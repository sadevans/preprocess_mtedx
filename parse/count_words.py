import yaml
from collections import defaultdict
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import spacy

# Загрузка данных из файла YAML
with open('/media/sadevans/T7/PERSONAL/Diplom/datsets/Vmeste/ru-ru/data/all/txt/all.yaml', 'r', encoding='utf-8') as file:
    data = yaml.safe_load(file)

# Создание словаря для хранения информации о словах для каждого спикера
speaker_words = defaultdict(list)

# Создание словарей для подсчета частоты слов и частей речи
word_frequency = defaultdict(int)
noun_frequency = defaultdict(int)
verb_frequency = defaultdict(int)
adjective_frequency = defaultdict(int)

nlp = spacy.load('ru_core_news_md')

# Обработка данных
for item in data:
    speaker_id = item['speaker_id']
    word = item['word']
    start = item['start']
    end = item['end']

    # Пропуск слишком коротких слов
    if len(word) > 1:
        # Добавление слова в список для каждого спикера
        speaker_words[speaker_id].append((word, start, end))

        # Подсчет частоты слов и частей речи
        word_frequency[word] += 1
        # pos_tags = pos_tag(word_tokenize(word, language='russian'))
        document = nlp(word)
        # for _, tag in document.pos_:
        for token in document:
            if token.pos_.startswith('NOUN'):
                noun_frequency[word] += 1
            elif token.pos_.startswith('VERB'):
                verb_frequency[word] += 1
            elif token.pos_.startswith('ADJ'):
                adjective_frequency[word] += 1

# Сортировка слов по частоте для каждого спикера
top_words_by_speaker = {speaker: sorted(words, key=lambda x: word_frequency[x[0]], reverse=True)[:30] for speaker, words in speaker_words.items()}

# Запись результатов в отдельные файлы YAML
with open('top_words_byspeakers.yaml', 'w', encoding='utf-8') as file:
    yaml.safe_dump(top_words_by_speaker, file, allow_unicode=True)

with open('freq_words.yaml', 'w', encoding='utf-8') as file:
    yaml.safe_dump(dict(word_frequency), file, allow_unicode=True)

with open('freq_nouns.yaml', 'w', encoding='utf-8') as file:
    yaml.safe_dump(dict(noun_frequency), file, allow_unicode=True)

with open('freq_verbs.yaml', 'w', encoding='utf-8') as file:
    yaml.safe_dump(dict(verb_frequency), file, allow_unicode=True)

with open('freq_adj.yaml', 'w', encoding='utf-8') as file:
    yaml.safe_dump(dict(adjective_frequency), file, allow_unicode=True)
