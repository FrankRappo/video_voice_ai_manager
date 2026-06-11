# A7 — Обновления документации

Этот файл собирает **все** правки в существующей документации, которые нужно
применить в рамках задачи A7 (миграция vision + correlator на Live API).

**Не применять сейчас.** Применять на последнем шаге A7 (Шаг 6), когда
миграция кода завершена и протестирована.

---

## 1. `tasks/A6_improve_video_analysis.md` — пометить устаревшую инструкцию

**Проблема:** строки 75-76 содержат ошибочную рекомендацию использовать
GA-модель `gemini-2.5-flash`, которая привела к текущему нарушению политики
моделей.

**Где править:** `tasks/A6_improve_video_analysis.md:75-76`

**Текущий текст:**
```markdown
Использовать `gemini-2.5-flash` (не native-audio) через Gemini API для LLM-задач
корреляции и классификации — дешевле и быстрее native-audio для текстовых задач.
```

**Заменить на:**
```markdown
> ⚠ **УСТАРЕЛО (см. A7).** Эта рекомендация была ошибочной: политика vvam
> требует использовать ТОЛЬКО `gemini-2.5-flash-native-audio-latest` через
> Live API (free, без rate limits). GA-модель `gemini-2.5-flash` не
> используется. См. [`gemini-models.md`](../gemini-models.md) и задачу A7
> (`tasks/A7_live_api_migration.md`).
>
> ~~Использовать `gemini-2.5-flash` (не native-audio) через Gemini API для
> LLM-задач корреляции и классификации — дешевле и быстрее native-audio для
> текстовых задач.~~
```

---

## 2. `tasks/A6_result.md` — пометка об ограничениях, решённых в A7

**Где править:** `tasks/A6_result.md:55-56` (раздел "Известные ограничения")

**Текущий текст:**
```markdown
1. **Rate limits Gemini Free Tier** — gemini-2.5-flash имеет лимит ~20 req/min, что приводит к 503/429 при анализе 57 кадров по отдельности. Решение: `correlate_with_images()` отправляет 15 ключевых кадров + транскрипцию в одном API-вызове. Для продакшн-использования рекомендуется платный API.
2. **Язык вывода** — при использовании `gemini-2.5-flash-lite` модель может отвечать на английском вместо русского. С `gemini-2.5-flash` ответы на русском.
```

**Заменить на:**
```markdown
1. **Rate limits Gemini Free Tier** — gemini-2.5-flash имеет лимит ~20 req/min, что приводит к 503/429 при анализе 57 кадров по отдельности. Решение в A6: `correlate_with_images()` отправляет 15 ключевых кадров + транскрипцию в одном API-вызове. **РЕШЕНО в A7:** весь трафик переведён на `gemini-2.5-flash-native-audio-latest` через Live API — preview модель без rate limits.
2. **Язык вывода** — при использовании `gemini-2.5-flash-lite` модель может отвечать на английском вместо русского. С `gemini-2.5-flash` ответы на русском. В A7 используется native-audio модель — язык стабильно русский.
```

---

## 3. `CHANGELOG.md` — новая запись о миграции

**Где править:** `CHANGELOG.md`, добавить **в начало** (перед записью от 2026-04-10).

**Текст для вставки** (дату и детали уточнить по факту выполнения A7):

```markdown
## <ДАТА A7>

### Migration: all Gemini traffic → Live API with native-audio preview

Completed task A7 — migrated `vision/gemini.py` and `core/correlator.py`
from GA model `gemini-2.5-flash` (via `generate_content`) to
`gemini-2.5-flash-native-audio-latest` (via Live API `bidiGenerateContent`).

**Why:** native-audio preview model is free and has no strict rate limits,
while GA `gemini-2.5-flash` on free tier caps at ~15-20 req/min and was
returning 429/503 during frame analysis. Per project policy (see
`gemini-models.md`) vvam uses **only** the preview native-audio model for
all Gemini traffic.

**What changed:**

- **New:** `core/gemini_live.py` — shared helper `call_live_multimodal()`
  for one-shot Live API calls with text + image parts.
- **`vision/gemini.py`**: `GeminiVision.analyze_frame` now opens a Live API
  session instead of `client.models.generate_content`. Default model
  changed to `gemini-2.5-flash-native-audio-latest`.
- **`core/correlator.py`**: both `_call_gemini` (text-only) and
  `_call_gemini_multimodal` (text + frames) rewritten to use Live API.
  Default model changed to `gemini-2.5-flash-native-audio-latest`.
- **`config.py`**: removed `gemini_model` field (the GA flash default).
  Only `gemini_audio_model = "gemini-2.5-flash-native-audio-latest"` remains.
- **Rate-limit retry:** kept exponential backoff on 503/429/UNAVAILABLE,
  though preview model should not produce these.

**Result on test video** `/home/hgff/work/IMG_7413.MP4`: see
`tasks/A7_result.md`. All 7 reference feedback items detected, including
the numeric-conflict case (WISH-5: tolerance 50% vs screen 15%). No rate
limit errors during the run.
```

---

## 4. `gemini-models.md` — обновить таблицу статуса

**Где править:** `gemini-models.md:46-51` (таблица "Текущее состояние").

**Текущая таблица:**
```markdown
| Компонент           | Файл                         | Модель (сейчас)                       | Endpoint          | Статус |
|---------------------|------------------------------|---------------------------------------|-------------------|--------|
| Транскрипция аудио  | `transcribers/gemini.py:42`  | `gemini-2.5-flash-native-audio-latest`| Live API          | ✅ OK   |
| Анализ кадров       | `vision/gemini.py:42`        | `gemini-2.5-flash` (GA, с лимитами)   | `generate_content`| ❌ NEED MIGRATION |
| Корреляция (A6)     | `core/correlator.py:49`      | `gemini-2.5-flash` (GA, с лимитами)   | `generate_content`| ❌ NEED MIGRATION |
```

**Заменить на:**
```markdown
| Компонент           | Файл                         | Модель                                | Endpoint          | Статус |
|---------------------|------------------------------|---------------------------------------|-------------------|--------|
| Транскрипция аудио  | `transcribers/gemini.py`     | `gemini-2.5-flash-native-audio-latest`| Live API          | ✅ OK   |
| Анализ кадров       | `vision/gemini.py`           | `gemini-2.5-flash-native-audio-latest`| Live API          | ✅ OK   |
| Корреляция          | `core/correlator.py`         | `gemini-2.5-flash-native-audio-latest`| Live API          | ✅ OK   |
| Live API helper     | `core/gemini_live.py`        | —                                     | Live API          | ✅ OK   |
```

**Также в `gemini-models.md`:**

- Убрать упоминание "Миграция vision + correlator на Live API запланирована
  в задаче A7" (строка 53) — заменить на "Все Gemini-вызовы унифицированы
  через helper `core/gemini_live.py` (задача A7, выполнено <ДАТА>)".
- В блоке "Модельная политика в config.py" (строка 111) убрать поле
  `gemini_model` (его больше нет в config.py).

---

## 5. Проверка по grep после применения всех правок

После применения всех обновлений выше — запустить:

```bash
cd /work/video_voice_ai_manager
# Не должно быть ни одного hit на GA-модель в Python-коде
grep -rn "gemini-2.5-flash[^-]" --include="*.py" .
# В документации допустимы упоминания только как "устарело" или в историческом контексте
grep -rn "gemini-2.5-flash[^-]" --include="*.md" .
```

**Ожидание:**
- `*.py` — 0 результатов
- `*.md` — только в: `gemini-models.md` (как "GA, не используется"),
  `CHANGELOG.md` (в исторических записях), `tasks/A6_*.md` (с пометкой
  "устарело"), `tasks/A7_*.md` (в контексте миграции).

---

## Чеклист применения

- [ ] `tasks/A6_improve_video_analysis.md:75-76` — пометить устаревшим
- [ ] `tasks/A6_result.md:55-56` — добавить пометку "решено в A7"
- [ ] `CHANGELOG.md` — добавить запись о A7 (в начало файла)
- [ ] `gemini-models.md:46-51` — обновить таблицу статуса (всё ✅)
- [ ] `gemini-models.md:53` — убрать упоминание "запланировано в A7"
- [ ] `gemini-models.md:111` — убрать `gemini_model` из блока config
- [ ] `grep "gemini-2.5-flash[^-]" --include="*.py"` — 0 результатов
- [ ] `grep "gemini-2.5-flash[^-]" --include="*.md"` — только допустимые
- [ ] Удалить этот файл (`tasks/A7_doc_updates.md`) после применения
