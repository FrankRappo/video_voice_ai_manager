# Задача A7: Миграция vision + correlator на Gemini Live API (native-audio preview)

## TL;DR

В vvam транскрипция уже работает на `gemini-2.5-flash-native-audio-latest`
(бесплатная preview, без rate limits). Но **vision** (`vision/gemini.py`) и
**correlator** (`core/correlator.py`) всё ещё используют GA-модель
`gemini-2.5-flash` через `client.models.generate_content` — это ошибка из A6.
GA-модель платная и имеет rate limits, может падать по 429/503.

**Нужно:** перевести vision и correlator на Live API с той же native-audio
preview-моделью, чтобы **весь** Gemini-трафик vvam шёл через единственную
бесплатную модель без лимитов.

> **Обязательно к прочтению перед началом:** [`gemini-models.md`](../gemini-models.md)
> — там описана политика моделей, причины выбора и пример правильного
> Live API вызова с мультимодальным контентом.

---

## Контекст

### Что уже есть

- **Транскрипция.** `transcribers/gemini.py::_transcribe_live` — работает,
  использует Live API правильно, отправляет аудио чанками с explicit VAD,
  собирает input+output транскрипцию. См. `CHANGELOG.md` (2026-04-10).
- **Транскрайбер знает про LIVE_ONLY:** есть константа `LIVE_ONLY_MODELS` в
  `transcribers/gemini.py:32-37`, переключатель
  `_is_live_model()` → `_transcribe_live` / `_transcribe_standard`.
- **Список моделей, которые можно использовать через Live API:**
  ```python
  LIVE_ONLY_MODELS = {
      "gemini-2.5-flash-native-audio-latest",
      "gemini-2.5-flash-native-audio-preview-09-2025",
      "gemini-2.5-flash-native-audio-preview-12-2025",
      "gemini-3.1-flash-live-preview",
  }
  ```

### Что НЕ соответствует политике

1. **`vision/gemini.py:42`** — default model `gemini-2.5-flash`, использует
   `client.models.generate_content` (строки 71-78) для анализа каждого кадра.
2. **`core/correlator.py:49`** — default model `gemini-2.5-flash`, использует
   `client.models.generate_content` для текстовой корреляции (строки 273-277)
   и для мультимодального вызова (строки 308-312).
3. **`config.py:26`** — `gemini_model: str = "gemini-2.5-flash"` — этот
   дефолт уходит в vision/correlator через factory. Его нужно либо удалить,
   либо поменять на native-audio.

### Почему нельзя просто заменить строку модели

Native audio модели — **LIVE-ONLY**. Они не принимают `generate_content`,
только `bidiGenerateContent` (Live API). Попытка вызвать GA-эндпоинт с
native-audio моделью вернёт ошибку. Значит **сам код вызова** нужно
переписать на `client.aio.live.connect(...)`.

---

## Цель

После задачи:
- [ ] `vision/gemini.py` вызывает Gemini через Live API session с
      `gemini-2.5-flash-native-audio-latest`
- [ ] `core/correlator.py` (оба метода `_call_gemini` и
      `_call_gemini_multimodal`) работают через Live API с той же моделью
- [ ] `config.py` более не содержит `gemini-2.5-flash` как дефолт
- [ ] Тест на `/home/hgff/work/IMG_7413.MP4` проходит без rate-limit ошибок
      и даёт результат сопоставимый с `tasks/hotelhack_client_feedback.md`
- [ ] `CHANGELOG.md` и `gemini-models.md` обновлены — таблица статуса
      помечена всеми зелёными галочками

---

## Шаги выполнения

> **Рекомендация:** работай поэтапно, запускай тесты после каждого шага.
> Не делай всё одним махом — Live API капризный, отлаживать проще
> инкрементально.

### Шаг 1 — Общий helper для Live API вызовов

Текущий `_transcribe_live_chunk` в `transcribers/gemini.py` сильно заточен
под аудио. Для vision/correlator нужен более общий helper.

**Варианты (выбери один, обоснуй выбор в коде):**

**A)** Вынести helper в новый модуль `core/gemini_live.py` с функцией:
```python
async def call_live_multimodal(
    api_key: str,
    model: str,
    text_prompt: str,
    images: list[tuple[bytes, str]] = None,  # (bytes, mime_type)
    audio_pcm: bytes = None,                  # 16kHz mono PCM, опционально
    system_instruction: str = None,
    response_modalities: list[str] = None,    # default ["TEXT"]
) -> str:
    """Одноразовый Live API вызов — открыть сессию, отправить контент,
    собрать текстовый ответ, закрыть сессию."""
```

**B)** Добавить метод `_call_live_multimodal` прямо в `GeminiTranscriber` и
вызывать его из vision/correlator через dependency injection.

**Предпочтителен вариант A** — меньше связности, transcriber остаётся
только про транскрипцию.

**Реализация helper'а:**

1. Открыть сессию:
   ```python
   client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
   config = types.LiveConnectConfig(
       response_modalities=response_modalities or ["TEXT"],
       system_instruction=types.Content(parts=[types.Part(text=system_instruction)]) if system_instruction else None,
   )
   async with client.aio.live.connect(model=model, config=config) as session:
       ...
   ```
2. Собрать `parts`: сначала изображения как `types.Part.from_bytes`,
   потом текстовый prompt как `types.Part(text=...)`.
3. Отправить через `session.send_client_content(turns=..., turn_complete=True)`.
4. Собрать ответ в цикле `async for msg in session.receive()`.
5. Вернуть конкатенированный текст.

> ⚠ **API `session.send_client_content` vs `send_realtime_input`:** для
> текста+изображений используется `send_client_content` (client_content
> сообщения). `send_realtime_input` — только для потоковой отправки audio.
> В транскрайбере используется `send_realtime_input`, потому что там PCM
> стрим; для vision/correlator это не подойдёт.
>
> Если в актуальной версии `google-genai` имена методов или типов
> изменились — смотри как устроено в `transcribers/gemini.py` и документацию
> https://ai.google.dev/gemini-api/docs/live

**Проверка шага 1:** написать мини-тест
`/work/video_voice_ai_manager/test_live_helper.py`, который:
- вызывает helper с одним текстовым промптом ("Say hello in Russian")
- вызывает helper с одной картинкой (любой frame из `/tmp/vvam_frames/`)
  и промптом "Describe this image in one sentence"
- печатает ответы, убеждается что оба не пустые

### Шаг 2 — Миграция `vision/gemini.py`

1. Импортировать helper из `core/gemini_live.py`.
2. Сменить дефолтную модель:
   ```python
   def __init__(self, api_key: str, model: str = "gemini-2.5-flash-native-audio-latest"):
   ```
3. Переписать `analyze_frame`: вместо `client.models.generate_content` →
   вызов helper'а с одним изображением. Сохранить retry (5 попыток,
   exponential backoff) на случай 503/429 — хотя preview модель таких
   ошибок давать не должна.
4. Метод `analyze_frames` переписывать не надо — он использует
   `analyze_frame` в цикле.

**Проверка шага 2:**
```bash
cd /work/video_voice_ai_manager
python3 -c "
import asyncio
from pathlib import Path
from vision.gemini import GeminiVision, CLIENT_FEEDBACK_PROMPT
import os

async def main():
    v = GeminiVision(api_key=os.environ['VVAM_GEMINI_API_KEY'])
    result = await v.analyze_frame(
        Path('/tmp/vvam_frames/frame_004.png'),
        prompt=CLIENT_FEEDBACK_PROMPT,
    )
    print(result[:500])

asyncio.run(main())
"
```

Результат должен быть валидный JSON с `ui_elements`, `numeric_values` и т.д.

### Шаг 3 — Миграция `core/correlator.py`

1. Импортировать helper из `core/gemini_live.py`.
2. Сменить дефолтную модель:
   ```python
   def __init__(self, api_key: str, model: str = "gemini-2.5-flash-native-audio-latest"):
   ```
3. Переписать `_call_gemini(prompt)` → вызов helper'а с только текстом.
4. Переписать `_call_gemini_multimodal(prompt, frames)` → вызов helper'а с
   изображениями + текстом. Обрати внимание: `frames` это
   `list[tuple[float, Path]]`, helper ожидает `list[tuple[bytes, str]]` —
   нужно прочитать файлы и определить mime.
5. Сохранить retry-логику и лимит на количество кадров.

**Проверка шага 3:** см. шаг 5 (финальный тест на видео).

### Шаг 4 — Обновить `config.py`

- Удалить поле `gemini_model` **или** переименовать в `gemini_legacy_model`
  и явно не использовать нигде.
- Убедиться что во всех factory (`transcribers/__init__.py`,
  `vision/__init__.py`) используется `gemini_audio_model`.
- Проверить что в `cli.py` и `web/` нет захардкоженных ссылок на
  `gemini-2.5-flash`:
  ```bash
  cd /work/video_voice_ai_manager
  grep -rn "gemini-2.5-flash[^-]" --include="*.py" .
  ```
  Все найденные места должны быть исправлены (либо на `native-audio-latest`,
  либо через config).

### Шаг 5 — Финальный тест на реальном видео

```bash
cd /work/video_voice_ai_manager
export VVAM_GEMINI_API_KEY=<ключ из окружения>
python3 cli.py video /home/hgff/work/IMG_7413.MP4 \
  --transcriber gemini --vision gemini \
  -f client-feedback -o /tmp/vvam_auto_feedback_A7.md \
  2>&1 | tee tasks/A7_output.log
```

**Критерии успеха:**
- [ ] Команда завершилась без ошибок (никаких 429/503/UNAVAILABLE в логе)
- [ ] Файл `/tmp/vvam_auto_feedback_A7.md` создан и непустой
- [ ] В нём есть **все 7 основных пунктов** из эталона
      `tasks/hotelhack_client_feedback.md` (БАГ-1, БАГ-2, ФИЧА-1 ... ФИЧА-5)
- [ ] **ФИЧА-5 (допуск 50% vs 15%)** обнаружена автоматически через
      `numeric_conflicts` — это главный тест корреляции
- [ ] Качество не хуже, чем в A6 (см. `tasks/A6_result.md`)

### Шаг 6 — Обновить документацию

1. **`CHANGELOG.md`** — добавить запись о A7 с указанием:
   - Какие файлы затронуты
   - Что теперь весь Gemini-трафик идёт через Live API
   - Что GA-модель `gemini-2.5-flash` полностью убрана
2. **`gemini-models.md`** — обновить таблицу статуса: все три строки
   должны быть ✅, столбец "Модель" — `gemini-2.5-flash-native-audio-latest`
   везде.
3. **`tasks/A7_result.md`** — создать отчёт по шаблону A6_result.md:
   - Что реализовано
   - Чеклист критериев успеха
   - Команда для повторного запуска
   - Путь к сгенерированному output'у
   - Известные ограничения (например, скорость: Live API сессия на каждый
     frame может быть медленнее чем `generate_content` — зафиксировать
     реальное время обработки)

---

## Риски и нюансы

### 1. Live API может быть медленнее на N отдельных вызовов

`generate_content` — это одноразовый HTTP запрос. Live API session — это
WebSocket с handshake. Если открывать новую сессию на **каждый** кадр (20
кадров), это может стоить 20× connection overhead.

**Митигация:** рассмотреть возможность **одной сессии на batch кадров**:
- Для vision: открыть одну сессию, отправить 20 user-turn'ов последовательно
  (каждый turn = один кадр + промпт), читать ответы.
- Для correlator: там и так одна multimodal-сессия, overhead небольшой.

Если это усложняет код — ok оставить "одна сессия на кадр", но замерить
и задокументировать в `A7_result.md` насколько медленнее.

### 2. Live API rate на preview модели — не гарантирован навсегда

Google может в любой момент ввести лимиты на preview-модели или убрать их.
**Митигация:** код должен иметь retry с backoff на 503/429 (как сейчас в
транскрайбере). Если retry исчерпан — логировать и падать с понятной
ошибкой, не прятать.

### 3. Multimodal ответ в текстовом виде

LiveConnectConfig умеет `response_modalities=["TEXT"]` или `["AUDIO"]`.
Для vision/correlator нужен **только текст** — проверить что при
`["TEXT"]` не возвращается аудио-поток и не надо декодить.

### 4. `wave` импорт в transcribers/gemini.py

В `transcribers/gemini.py:8` уже импортирован `wave` — используется для
чтения PCM. Для vision/correlator wave не нужен, helper не должен тянуть
эту зависимость.

### 5. Не коммитить API-ключи

- API ключ **только через env var** `VVAM_GEMINI_API_KEY`, см. раздел
  "Как запускать" в A6.
- Перед коммитом проверить:
  ```bash
  cd /work/video_voice_ai_manager && git diff --cached | grep -i "aizas\|api_key.*=.*['\"]" || echo "clean"
  ```

---

## Файлы, которые надо изменить / создать

- [ ] `core/gemini_live.py` — **НОВЫЙ** helper для Live API вызовов
      (text-only и multimodal)
- [ ] `vision/gemini.py` — переход на Live API, смена default model
- [ ] `core/correlator.py` — переход на Live API, смена default model
- [ ] `config.py` — убрать/исправить `gemini_model` дефолт
- [ ] `test_live_helper.py` — **НОВЫЙ** sanity test для helper'а
- [ ] `CHANGELOG.md` — запись о миграции
- [ ] `gemini-models.md` — обновить таблицу статуса
- [ ] `tasks/A7_result.md` — **НОВЫЙ** финальный отчёт

---

## Как запускать эту задачу

1. Прочитать `gemini-models.md` целиком
2. Прочитать текущие `transcribers/gemini.py`, `vision/gemini.py`,
   `core/correlator.py` — понять структуру
3. Выполнять шаги 1-6 последовательно, проверять каждый шаг отдельным тестом
4. В конце — финальный e2e тест на видео и отчёт

**API ключ:** через env `VVAM_GEMINI_API_KEY` (уже установлена).

**Тестовые данные:** `/home/hgff/work/IMG_7413.MP4`, эталон —
`tasks/hotelhack_client_feedback.md`.

**Запускать с:** `cd /work/video_voice_ai_manager`.
