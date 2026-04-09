# Обработка видео через Claude

## Проблема

Claude не умеет смотреть видео напрямую. Но умеет читать изображения и текст. Нужны посредники, которые разложат видео на понятные части.

---

## Два канала информации из видео

### 1. Визуальный ряд -> кадры (скриншоты)

Извлечь кадры из видео через ffmpeg, затем Claude анализирует их как изображения.

```bash
# Каждую секунду по кадру
ffmpeg -i video.mp4 -vf "fps=1" frame_%04d.png

# Каждые 5 секунд (для длинных видео)
ffmpeg -i video.mp4 -vf "fps=1/5" frame_%04d.png

# Каждые 10 секунд
ffmpeg -i video.mp4 -vf "fps=1/10" frame_%04d.png

# Только ключевые кадры (при смене сцены) — самый экономный вариант
ffmpeg -i video.mp4 -vf "select=gt(scene\,0.3)" -vsync vfp frame_%04d.png
```

Выбор частоты зависит от типа видео:
- Скринкаст/презентация: 1 кадр в 5-10 сек (экран меняется редко)
- Живая съёмка: 1 кадр в 1-2 сек
- Детекция сцен: лучший вариант, когда видео длинное и неравномерное

### 2. Звуковая дорожка -> текст

Извлечь аудио, транскрибировать, затем Claude читает текст.

#### Шаг 1: Извлечь аудио

```bash
ffmpeg -i video.mp4 -vn -acodec pcm_s16le -ar 16000 audio.wav
```

#### Шаг 2: Транскрибировать

**Whisper (OpenAI)** — лучший вариант для распознавания речи:

```bash
# Установка
pip install openai-whisper

# Транскрипция (русский язык)
whisper audio.wav --language ru --model medium --output_format txt

# Модели (точность / скорость):
#   tiny    — быстрый, низкая точность
#   base    — быстрый, приемлемая точность
#   small   — средний баланс
#   medium  — хорошая точность (рекомендуется для русского)
#   large   — лучшая точность, медленный
```

**faster-whisper** — оптимизированная версия, быстрее в 4-6 раз:

```bash
pip install faster-whisper

# Использование из Python:
from faster_whisper import WhisperModel
model = WhisperModel("medium", compute_type="int8")
segments, info = model.transcribe("audio.wav", language="ru")
for segment in segments:
    print(f"[{segment.start:.1f}s - {segment.end:.1f}s] {segment.text}")
```

---

## Полный пайплайн

```
Видео (video.mp4)
  │
  ├── ffmpeg ──→ кадры (PNG) ──→ Claude читает изображения
  │                                (Read tool для каждого кадра)
  │
  └── ffmpeg ──→ аудио (WAV) ──→ Whisper ──→ текст (TXT) ──→ Claude читает текст
```

### Автоматизация (один скрипт)

```bash
#!/bin/bash
VIDEO="$1"
OUTDIR="./video_analysis"
mkdir -p "$OUTDIR/frames"

echo "=== Извлечение кадров ==="
ffmpeg -i "$VIDEO" -vf "fps=1/5" "$OUTDIR/frames/frame_%04d.png" -y 2>/dev/null
echo "Кадров: $(ls "$OUTDIR/frames/" | wc -l)"

echo "=== Извлечение аудио ==="
ffmpeg -i "$VIDEO" -vn -acodec pcm_s16le -ar 16000 "$OUTDIR/audio.wav" -y 2>/dev/null

echo "=== Транскрипция ==="
whisper "$OUTDIR/audio.wav" --language ru --model medium --output_format txt --output_dir "$OUTDIR" 2>/dev/null

echo "=== Готово ==="
echo "Кадры: $OUTDIR/frames/"
echo "Текст: $OUTDIR/audio.txt"
```

---

## Что подходит для каких задач

| Тип видео | Кадры | Аудио | Комментарий |
|-----------|-------|-------|-------------|
| Скринкаст с голосом | Да (fps=1/5) | Да | Оба канала важны |
| Презентация | Да (смена сцен) | Да | Слайды + речь |
| Голосовое ТЗ (без экрана) | Нет | Да | Достаточно транскрипции |
| Демо интерфейса (без звука) | Да (fps=1) | Нет | Только кадры |
| Длинное обучающее видео | Да (fps=1/10) | Да | Экономим на кадрах |

---

## Требования

- **ffmpeg**: обычно уже установлен, иначе `apt install ffmpeg`
- **whisper**: `pip install openai-whisper` (нужен Python 3.8+)
- **faster-whisper**: `pip install faster-whisper` (быстрее, меньше RAM)
- **GPU**: не обязателен, но ускоряет Whisper в 5-10 раз (CUDA)
- **Диск**: ~1 MB на кадр (PNG), модель Whisper medium ~1.5 GB
