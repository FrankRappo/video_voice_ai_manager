# Политика моделей Gemini в vvam

> Короткое правило: **в vvam используется ТОЛЬКО preview native-audio модель Gemini**
> (`gemini-2.5-flash-native-audio-latest`). GA-модели (`gemini-2.5-flash`,
> `gemini-2.5-pro` и т.п.) **не используются** — у них есть rate limits на free
> tier, а preview-модель бесплатная и без ограничений.

---

## Почему только preview native-audio?

1. **Free / без лимитов.** GA-модели (`gemini-2.5-flash`) на free tier
   ограничены ~15 запросов/мин и могут возвращать 429/503 под нагрузкой.
   `gemini-2.5-flash-native-audio-latest` — preview, бесплатная, без жёстких
   лимитов на момент 2026-04.
2. **Лучшее качество транскрипции.** Native audio модель напрямую понимает
   речь без промежуточного STT — это даёт более точные таймкоды и не теряет
   слова. См. `CHANGELOG.md` (запись от 2026-04-10): переход на native-audio
   поднял покрытие транскрипции с ~22% до ~97% на тестовом аудио 4:46.
3. **Единый контекст для мультимодального анализа.** Одна Live API сессия
   может принимать audio + images + prompt в одном потоке — это даёт
   correlator'у целостное видение, а не три изолированных вызова.

## Почему это сложнее, чем просто сменить строку в config

Native-audio модели — **LIVE-ONLY**. Они работают **только** через
`bidiGenerateContent` (Live API), а **не** через обычный
`client.models.generate_content`. Список LIVE_ONLY моделей зафиксирован в
`transcribers/gemini.py`:

```python
LIVE_ONLY_MODELS = {
    "gemini-2.5-flash-native-audio-latest",
    "gemini-2.5-flash-native-audio-preview-09-2025",
    "gemini-2.5-flash-native-audio-preview-12-2025",
    "gemini-3.1-flash-live-preview",
}
```

Попытка передать LIVE_ONLY модель в `generate_content` вернёт ошибку API. То
есть **любой код, использующий Gemini в vvam, должен идти через Live API
session**, а не через одноразовый `generate_content`.

## Текущее состояние (на 2026-04-10)

| Компонент           | Файл                         | Модель (сейчас)                       | Endpoint          | Статус |
|---------------------|------------------------------|---------------------------------------|-------------------|--------|
| Транскрипция аудио  | `transcribers/gemini.py:42`  | `gemini-2.5-flash-native-audio-latest`| Live API          | ✅ OK   |
| Анализ кадров       | `vision/gemini.py:42`        | `gemini-2.5-flash` (GA, с лимитами)   | `generate_content`| ❌ NEED MIGRATION |
| Корреляция (A6)     | `core/correlator.py:49`      | `gemini-2.5-flash` (GA, с лимитами)   | `generate_content`| ❌ NEED MIGRATION |

Миграция vision + correlator на Live API запланирована в **задаче A7**
(`tasks/A7_live_api_migration.md`).

## Что делать при добавлении нового Gemini-вызова

**ВСЕГДА** используй Live API через helper, а не `client.models.generate_content`.
Общая схема:

```python
from google import genai
from google.genai import types

client = genai.Client(
    api_key=api_key,
    http_options={"api_version": "v1alpha"},  # обязательно для Live API
)

config = types.LiveConnectConfig(
    response_modalities=["TEXT"],  # или ["AUDIO"] если надо аудио-ответ
    system_instruction=types.Content(parts=[types.Part(text="...")]),
)

async with client.aio.live.connect(
    model="gemini-2.5-flash-native-audio-latest",
    config=config,
) as session:
    # Отправка мультимодального контента
    parts = []
    for img_bytes, mime in images:
        parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))
    parts.append(types.Part(text=prompt))

    await session.send_client_content(
        turns=types.Content(role="user", parts=parts),
        turn_complete=True,
    )

    # Сбор ответа
    chunks = []
    async for msg in session.receive():
        if msg.text:
            chunks.append(msg.text)
        if msg.server_content and msg.server_content.turn_complete:
            break

    return "".join(chunks)
```

> **Важно:** конкретный API сбора ответа зависит от версии `google-genai` —
> детали сверяй с актуальной документацией и уже работающим кодом в
> `transcribers/gemini.py::_transcribe_live_chunk`.

## Документация Google

- Live API overview: https://ai.google.dev/gemini-api/docs/live
- Native audio models: https://ai.google.dev/gemini-api/docs/models/gemini#native-audio
- Модели и лимиты: https://ai.google.dev/gemini-api/docs/rate-limits

## Модельная политика в config.py

```python
# Gemini settings — ВСЁ должно быть на native-audio preview
gemini_model: str = "gemini-2.5-flash"                          # ❌ legacy, удалить в A7
gemini_audio_model: str = "gemini-2.5-flash-native-audio-latest" # ✅ правильно
```

После выполнения A7 поле `gemini_model` будет удалено (или будет указывать
на ту же native-audio модель) — чтобы исключить случайное использование
GA-модели.
