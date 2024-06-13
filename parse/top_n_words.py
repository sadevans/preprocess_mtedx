import yaml

# Загрузка данных из файла YAML
with open('/home/sadevans/space/personal/preprocess_mtedx/parse/freq_words.yaml', 'r', encoding='utf-8') as file:
    data = yaml.safe_load(file)

# Сортировка слов по их популярности
sorted_words = sorted(data.items(), key=lambda x: x[1], reverse=True)

# Вывод самых популярных 100 слов
top_100_words = dict(sorted_words[:200])
print(top_100_words)
