# Заметка использования (2026-05-28)

> Указание владельца от 04.11.2026: длительные задания теперь всегда
> запускаются в фоне через `python3 run_background.py voice ...`.
> Примеры ниже сохранены как история прямого CLI. Актуальные правила:
> [BACKGROUND.md](BACKGROUND.md), [AGENTS.md](AGENTS.md).

## Команда для транскрипции голосового через Gemini

```bash
cd /work/video_voice_ai_manager && python3 cli.py voice <PATH_TO_OGG> -o <OUTPUT.md>
```

Пример:

```bash
cd /work/video_voice_ai_manager && python3 cli.py voice /work/tg/files/1462_2026-05-28_13-02-27_voice.ogg -o /tmp/voice_1462.md
```

По умолчанию используется `--transcriber gemini` (модель `gemini-2.5-flash-native-audio-latest` через Live API), формат `markdown`. Ключ из `config.py` / переменной окружения подхватывается автоматически.

## Наблюдения

- **Транзиентные ошибки Live API**: первые 2 попытки упали с `APIError 1006 abnormal closure` и `1011 deadline expired`. Третий запуск (без изменений) — успех. Подтверждает пункт 2 из `USAGE_FEEDBACK_2026-04-23.md` — встроенного retry до сих пор нет.
- **Альтернативные транскрайберы недоступны**:
  - `--transcriber whisper-local` → `ModuleNotFoundError: No module named 'whisper'`
  - `--transcriber openai` → `ImportError: Install openai`
  - Фактически единственный рабочий путь — Gemini, и при сетевых проблемах нужно просто пере-запускать команду вручную.
