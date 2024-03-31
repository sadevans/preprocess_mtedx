# Подготовка датасета mTEdx-ru для обучения модели AUTO-AVSR

# Шаг 1. Загрузка датасета

Для того, чтобы скачать датасет, необходимо запустить файл `download.py` с некоторыми аргументами:

```bash
python3 download.py --dataset mtedx --root-path your/path/to/download/folder --src-lang ru
```
Обязательные аргументы:
- `--dataset` - название датасета
- `--root-path` - путь до местоположения загрузки датасета
- `--src-lang` - язык датасета

Необязательные аргументы:
- `--download` - требуется ли загрузка сжатого датасета. По умолчанию - True, однако если у вас уже скачан датасет, и вы хотите скачать только видео - передайте `--download 0`
- `--num-workers` - количество параллельных процессов


# Шаг 2. Предобработка датасета для обрезки видео

# Шаг 3. Обрезка ROI губ

# Шаг 4. Создание SentencePiece модели