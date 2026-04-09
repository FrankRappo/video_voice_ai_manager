# Задача: Написать output форматтеры и CLI

## Контекст
Проект video_voice_ai_manager — универсальный AI-powered анализатор видео и голосовых.
Код в /work/video_voice_ai_manager/. Уже написаны:
- config.py — конфигурация (Config dataclass с load/save/validate)
- transcribers/base.py — Segment(start, end, text), TranscriptionResult(segments, language, full_text)
- vision/base.py — FrameAnalysis(timestamp, description, frame_path)
- transcribers/ — gemini.py, whisper_local.py, openai_api.py
- vision/ — gemini.py, openai_api.py, ollama.py

Параллельно другой агент пишет core/ модули (video.py, voice.py, dictate.py, screenshot.py).

ОБЯЗАТЕЛЬНО прочитай base.py файлы и config.py перед началом.

## Что сделать

### 1. output/markdown.py
Markdown форматтер:
- Функция format_markdown(transcript: TranscriptionResult, frames: list[FrameAnalysis], metadata: dict) -> str
- Заголовок с метаданными (источник, длительность, бэкенды)
- Раздел "Транскрипция" — таймкоды с текстом: `[00:15 - 00:32] Текст...`
- Раздел "Кадры" — таймкод + описание + путь к файлу кадра
- Раздел "Саммари" если есть
- Pipe-friendly (работает в любом терминале)

### 2. output/json_out.py
JSON форматтер:
- Функция format_json(transcript, frames, metadata) -> str
- Полная структура: metadata, segments[], frames[], full_text
- Пригоден для потребления любым API/агентом

### 3. output/srt.py
SRT субтитры:
- Функция format_srt(transcript) -> str
- Стандартный SRT: нумерация, таймкоды HH:MM:SS,mmm, текст

### 4. output/formatters.py
Фабрика форматтеров:
- get_formatter(name: str) -> callable
- Поддержка: "markdown", "json", "srt"
- Raise ValueError для неизвестных форматов

### 5. cli.py — CLI точка входа
argparse CLI с подкомандами:

```
vvam video <source> [options]    — анализ видео (файл или URL)
vvam voice <source> [options]    — анализ голосовых
vvam dictate [options]           — надиктовка
vvam screenshot <source> [options] — извлечение кадров
vvam server [options]            — запуск веб-сервера
```

Опции video:
--transcriber (gemini|whisper-local|openai), --vision (gemini|openai|ollama)
--output/-o (путь к файлу), --format/-f (markdown|json|srt)
--chunk (минуты), --audio-only, --from, --to, --fps, --prompt

Опции voice:
--transcriber, --format, --output, --all (пакетная обработка папки)

Опции dictate:
--file (аудиофайл), --mic (запись с микро), --transcriber

Опции screenshot:
--time (таймкод), --from, --to, --fps

Опции server:
--host, --port

Глобальные:
--config (путь к конфигу), --gemini-key, --openai-key

Entry point: функция main() для console_scripts в setup.py
Pipe-friendly: если stdout не tty — без цветов и прогресса, чистый вывод.

## Как проверить
```bash
cd /work/video_voice_ai_manager
python3 -c "from output.formatters import get_formatter; print('OK')"
python3 cli.py --help
python3 cli.py video --help
python3 cli.py voice --help
```

## Файлы для изменения
- /work/video_voice_ai_manager/output/markdown.py (создать)
- /work/video_voice_ai_manager/output/json_out.py (создать)
- /work/video_voice_ai_manager/output/srt.py (создать)
- /work/video_voice_ai_manager/output/formatters.py (создать)
- /work/video_voice_ai_manager/cli.py (создать)
