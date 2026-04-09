# Задача: Написать FastAPI сервер, веб-интерфейс, setup.py

## Контекст
Проект video_voice_ai_manager — универсальный AI-powered анализатор видео и голосовых.
Код в /work/video_voice_ai_manager/. Уже написаны:
- config.py — конфигурация
- transcribers/ — base.py + gemini.py, whisper_local.py, openai_api.py
- vision/ — base.py + gemini.py, openai_api.py, ollama.py

Параллельно другие агенты пишут:
- core/ (video.py, voice.py, dictate.py, screenshot.py)
- output/ (markdown.py, json_out.py, srt.py, formatters.py) и cli.py

ОБЯЗАТЕЛЬНО прочитай существующие файлы перед началом.

## Что сделать

### 1. web/server.py — FastAPI сервер
- POST /api/video — загрузка видео файла или отправка URL, возвращает анализ
- POST /api/voice — загрузка голосового, возвращает транскрипцию
- POST /api/screenshot — извлечь кадр по таймкоду, вернуть изображение
- GET /api/status/{job_id} — статус длинной задачи
- WebSocket /ws/progress — стриминг прогресса
- Serve static файлы и веб UI
- CORS для любого origin
- Загруженные файлы сохраняются во временную папку
- Background tasks для длинных видео

### 2. web/templates/index.html — веб-интерфейс
Один HTML файл с встроенным CSS и JS (vanilla, без фреймворков):
- Тёмная современная тема
- Зона загрузки: drag & drop или выбор файла для видео/аудио
- Поле ввода URL (вставить ссылку YouTube и тд)
- Встроенный видеоплеер
- Таймлайн с кликабельными таймкодами
- Панель транскрипции синхронизированная с видео (клик на текст → видео перематывает)
- Панель с превью кадров и таймкодами
- Поиск по транскрипции
- Кнопки экспорта: скачать .md, .json, .srt
- Адаптивная вёрстка (работает на мобильных)
- Индикатор прогресса через WebSocket
- Выбор бэкендов (transcriber/vision) в настройках

### 3. requirements.txt
```
google-genai>=1.0.0
fastapi>=0.100.0
uvicorn>=0.20.0
python-multipart>=0.0.5
aiofiles>=23.0.0
```
Опциональные (в комментариях):
```
# pip install openai          # для OpenAI backend
# pip install faster-whisper   # для локального Whisper
# pip install openai-whisper   # альтернатива faster-whisper
# pip install yt-dlp           # для скачивания видео по URL
# pip install sounddevice      # для записи с микрофона
```

### 4. setup.py
```python
setup(
    name="video-voice-ai-manager",
    version="0.1.0",
    description="Universal AI-powered video & voice analyzer",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[...core deps...],
    extras_require={
        "openai": ["openai"],
        "whisper": ["faster-whisper"],
        "web": ["fastapi", "uvicorn", "python-multipart", "aiofiles"],
        "all": [...everything...],
    },
    entry_points={"console_scripts": ["vvam=cli:main"]},
)
```

### 5. pyproject.toml
Современный Python packaging config, дублирующий setup.py.

## Как проверить
```bash
cd /work/video_voice_ai_manager
python3 -c "from web.server import app; print('OK')"
pip install -e . 2>/dev/null || echo "setup.py check"
cat requirements.txt
```

## Файлы для изменения
- /work/video_voice_ai_manager/web/server.py (создать)
- /work/video_voice_ai_manager/web/templates/index.html (создать)
- /work/video_voice_ai_manager/requirements.txt (создать)
- /work/video_voice_ai_manager/setup.py (создать)
- /work/video_voice_ai_manager/pyproject.toml (создать)
