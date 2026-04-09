# Задача: Написать core модули для VVAM

## Контекст
Проект video_voice_ai_manager — универсальный AI-powered анализатор видео и голосовых сообщений.
Код лежит в /work/video_voice_ai_manager/. Уже написаны:
- config.py — конфигурация
- transcribers/base.py — интерфейс транскрайберов (Segment, TranscriptionResult, BaseTranscriber)
- transcribers/gemini.py, whisper_local.py, openai_api.py — реализации
- vision/base.py — интерфейс vision (FrameAnalysis, BaseVision)
- vision/gemini.py, openai_api.py, ollama.py — реализации

ОБЯЗАТЕЛЬНО прочитай эти файлы перед началом работы, чтобы использовать правильные интерфейсы.

## Что сделать

Написать 4 файла в /work/video_voice_ai_manager/core/:

### 1. core/video.py — режим разбора видео
- Класс VideoAnalyzer
- Принимает видео файл ИЛИ URL (YouTube и тд)
- Если URL → скачать через yt-dlp (subprocess)
- Извлечь кадры через ffmpeg: поддержка fps и scene detection
- Извлечь аудио через ffmpeg (wav 16kHz)
- Транскрибировать аудио через transcriber backend (из transcribers/)
- Проанализировать кадры через vision backend (из vision/)
- Для длинных видео (>10 мин): нарезать на чанки, обработать каждый, склеить
- Поддержка --audio-only (без кадров, экономия токенов)
- Поддержка --from/--to (временной диапазон)
- Возвращает структурированный результат: TranscriptionResult + list[FrameAnalysis] + metadata

### 2. core/voice.py — режим голосовых сообщений
- Класс VoiceAnalyzer
- Принимает файл или директорию (пакетная обработка)
- Форматы: ogg, opus, mp3, wav, m4a, flac, aac, wma
- Конвертация в wav через ffmpeg если нужно
- Транскрипция через transcriber backend
- Авто-определение Telegram (.ogg) и WhatsApp (.opus)
- Возвращает TranscriptionResult

### 3. core/dictate.py — режим надиктовки
- Класс Dictator
- Принимает аудиофайл
- Транскрибирует и выдаёт чистый текст
- Pipe-friendly вывод (для пайпа в любой AI CLI агент)
- Опциональная запись с микрофона через sounddevice (try/except если нет)

### 4. core/screenshot.py — извлечение скриншотов
- Класс ScreenshotExtractor
- Извлечь один кадр по таймкоду из видео
- Извлечь диапазон кадров (from-to с заданным fps)
- Имена файлов с таймкодами: 00m_05s.png, 01m_23s.png

Все модули:
- async (asyncio)
- Используют subprocess для ffmpeg/yt-dlp
- Кроссплатформенные (Linux/macOS/Windows)
- Используют интерфейсы из transcribers/base.py и vision/base.py

## Как проверить
```bash
cd /work/video_voice_ai_manager
python3 -c "from core.video import VideoAnalyzer; print('OK')"
python3 -c "from core.voice import VoiceAnalyzer; print('OK')"
python3 -c "from core.dictate import Dictator; print('OK')"
python3 -c "from core.screenshot import ScreenshotExtractor; print('OK')"
```

## Файлы для изменения
- /work/video_voice_ai_manager/core/video.py (создать)
- /work/video_voice_ai_manager/core/voice.py (создать)
- /work/video_voice_ai_manager/core/dictate.py (создать)
- /work/video_voice_ai_manager/core/screenshot.py (создать)
