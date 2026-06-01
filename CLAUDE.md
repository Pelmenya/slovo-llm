# slovo-llm — заметки для Claude

Контекст и эмпирика по этому проекту, чтобы будущие сессии не разбирались с нуля.
README остаётся официальной документацией; здесь — то, что узнали в процессе работы.

## Co-agents coordination (Layer 1)

Ты — агент **slovo-llm-runtime**. Параллельно в смежных репах могут идти другие Claude Code сессии.

**Shared board:** `C:\Users\Diamond\.claude\AGENT-STATUS.md` — единая точка координации.
**Setup doc:** `C:\Users\Diamond\Desktop\multi-agent-setup\multi-agent-setup.md`.

### Sibling agents

| Агент | Репо | Точки касания с slovo-llm-runtime |
|---|---|---|
| **slovo-backend** | `slovo/` | основной потребитель Ollama API на `localhost:11434` (через абстракцию `libs/llm`); A/B-сравнение laguna vs Claude/OpenAI |

### Protocol

**Перед задачей:**
1. Прочитать `~/.claude/AGENT-STATUS.md`
2. Если slovo-backend сейчас гоняет тяжёлый bench/extraction (water-analysis ETL, vision-catalog) и ты собираешься рестартить `ollama-laguna` / менять модель / KV-cache / NUM_PARALLEL → **спросить у пользователя**, не запускаться (порвёшь живые запросы)
3. Добавить строку про себя в `## Active` (Agent / Repo / Started / Intent / Touching / ETA / Notes)

**Во время работы:** обновлять intent при milestone'ах.

**После задачи:**
- Перенести строку из `## Active` в `## Completed`
- Если менял модель / quantization / контекст / KV-cache / API surface (новый endpoint Ollama, изменение `KEEP_ALIVE`, переезд на llama.cpp) → handoff `slovo-llm-runtime → slovo-backend` в `## Recent handoffs` с numbers (tok/s, VRAM, latency) и breaking-change'ами для prompts

**User (Дмитрий) = mediator on conflicts. Auto-merge cross-repo запрещён.**

## Субагенты (Code Review + специалисты)

В `.claude/agents/` установлены 4 субагента из [wshobson/agents](https://github.com/wshobson/agents) под профиль репы (Docker + GPU + Ollama tuning). **При вызове Agent tool используй `subagent_type` из таблицы ниже, а не `general-purpose`.**

| Агент | Когда использовать | subagent_type |
|---|---|---|
| **debugger** | Crash контейнера, GPU не пробрасывается, Ollama unhealthy, странности с layer offload | `debugger` |
| **performance-engineer** | Бенчи tok/s, prompt_eval rate, выбор quantization (q4_0 / q8_0 / f16), NUM_PARALLEL, KV-cache tuning | `performance-engineer` |
| **deployment-engineer** | `docker-compose.yml`, GPU passthrough через `deploy.resources.devices`, volumes, env Ollama (`KEEP_ALIVE`, `OLLAMA_FLASH_ATTENTION`), Docker Desktop апгрейды | `deployment-engineer` |
| **observability-engineer** | nvidia-smi мониторинг, GPU/VRAM/power logging, structured логи Ollama, алёрты по утилизации | `observability-engineer` |
| **docs-reviewer** | Дрейф документации: CLAUDE.md бенчмарки vs `stress-test-output.md`, runtime конфиг (NUM_PARALLEL, KV-cache) vs `docker-compose.yml`, Ollama version, история инфры на противоречия с current state | `docs-reviewer` |

| Команда пользователя | subagent_type |
|---|---|
| «не работает / зависло / OOM» | `debugger` |
| «прогнать бенч» / «оптимизировать модель» | `performance-engineer` |
| «настроить docker-compose» / «GPU не виден» | `deployment-engineer` |
| «настрой мониторинг» / «алерты на VRAM» | `observability-engineer` |
| «проверь документацию» / «бенчи актуальны?» | `docs-reviewer` |

Все агенты — модель `opus`. Малая репа, без pre-PR ритуала — вызывать по ситуации. **После каждого нового stress-test прогона** / изменения `docker-compose.yml` — стоит звать `docs-reviewer`, потому что бенчмарки и runtime-конфиг в CLAUDE.md дрейфуют первыми.

## Железо и конфиг (резюме)

- RTX 4070 Ti SUPER 16 GB VRAM, i9-11900K, 64 GB RAM, Windows 10 Pro + Docker Desktop / WSL2
- Ollama 0.23+ в контейнере `ollama-laguna`, порт 11434
- Модель: `laguna-xs.2:q4_K_M` (~21.5 GiB), volume `slovo-llm_ollama_data`
- Контекст 32K (пользователь явно сказал "32 маловато" — НЕ предлагать резать), KV-cache q8_0, NUM_PARALLEL=1, Flash Attention on, KEEP_ALIVE 24h

## Бенчмарки q4_K_M @ 32K (2026-05-09)

### NUM_PARALLEL=2, KV q8_0 (baseline)

| Метрика | Значение |
|---|---|
| Generation | **10.8 tok/s** |
| Prompt eval | **301 tok/s** (763 / 2.5 с) |
| Layer offload | 24/41 GPU, 17 CPU |
| VRAM | 15.9 / 16 GiB |
| GPU util | 30–40% |
| GPU power | 40 Вт из 285 Вт |

### NUM_PARALLEL=1, KV q8_0

| Метрика | Значение | Δ vs baseline |
|---|---|---|
| Generation | 11.1 tok/s | +3% |
| Layer offload | 25/41 GPU, 16 CPU | +1 слой |
| KvSize | 32768 | -50% |

### NUM_PARALLEL=1, KV q4_0

| Метрика | Значение | Δ vs baseline |
|---|---|---|
| Generation | 11.3 tok/s | +5% |
| Prompt eval | 333 tok/s | +11% |
| Layer offload | 25/41 GPU, 16 CPU | +1 слой |

### NUM_PARALLEL=1, KV f16 (без квантования)

| Метрика | Значение | Δ vs baseline |
|---|---|---|
| Generation | 10.9 tok/s | +1% |
| Prompt eval | 320 tok/s | +6% |
| Layer offload | **24/41** GPU, 17 CPU | -1 слой |

### Финальное решение: NUM_PARALLEL=1, KV q8_0

Разница q4_0 / q8_0 / f16 по скорости — **в пределах шума одного прогона** (10.9–11.3 tok/s). Слоёв на GPU при q8_0 столько же, сколько при q4_0 (25), а при f16 — на один меньше (24). Качество выхода во всех трёх случаях имеет баги одного класса.

→ q8_0 — лучший баланс: качество выше чем у q4_0 на длинных контекстах, без проигрыша по скорости и слоям.

**Главный вывод про bottleneck:** GPU простаивает (40 Вт из 285), потому что слои на CPU тормозят пайплайн — каждый токен идёт через GPU → CPU → GPU. Это **не проблема видеокарты, а проблема разделения**.

**Почему оптимизации дали мало:** Ollama резервирует значительную часть освободившейся VRAM под compute buffers (working memory для attention/gating, особенно велики для MoE с 256 экспертами и роутингом). Линейная экстраполяция "1 слой ≈ 500 MiB VRAM" не работает.

- `NUM_PARALLEL: 2 → 1` освободила ~2.5 GiB → +1 слой на GPU (ожидалось +5–6)
- `KV q8_0 → q4_0` освободила ещё ~1.25 GiB → 0 слоёв (Ollama пыталась дотащить 26-й, не хватило ~немного MiB), но per-token compute на q4_0 KV чуть быстрее

**Что ещё могло бы помочь, но не делали:**
- Уменьшение контекста ниже 32K — отвергнуто пользователем (32K — нижняя граница для агентских сессий).
- Прямой `llama.cpp` без Ollama — позволяет тоньше настраивать VRAM, но переписывание всей инфры.
- Ждать релизов Ollama, где улучшат VRAM-планирование для MoE.

**Качество на q4_0 KV-cache:** на той же агентской задаче (~2.5K токенов выхода) баги такого же класса (тот же extra closing paren в том же месте). Никаких заметных артефактов от q4_0 в коротких ответах не наблюдается. На длинных контекстах (>20K) теоретически может быть просадка, но не тестировалось — будет видно в реальной работе.

## Качество вывода (laguna-xs.2 q4_K_M)

На реалистичной агентской задаче (2.3K токенов выхода, NestJS-провайдер с retry/streaming/типизированными ошибками):

- **Архитектурно попадает в цель**: правильная иерархия классов ошибок, ConfigService, Logger, undici Agent через factory provider, AsyncGenerator для стрима, парсинг NDJSON через ReadableStream + TextDecoder, строгие типы без `any`.
- **Но есть реальные баги, не пройдут `tsc`**:
  - Лишние скобки/фигурные скобки в случайных местах (синтаксис ломается)
  - Смешивает `return` и `yield` в одном методе → не компилируется
  - Использует несуществующие опции API (`fetch({ agent })` вместо `dispatcher`, `data.prompt_eval` вместо `prompt_eval_count`)
  - Странные TypeScript union'ы

**Вывод:** уровень "хороший черновик / pair programming". НЕ автономный агент, мерджащий PR'ы — каждый ответ требует ручного review. Sonnet/Opus такого качества бы не дали.

## Куда подходит / не подходит

**Подходит:**
- Скаффолдинг и черновики с обязательным человеком в петле
- Локальный inference на чувствительном коде, который нельзя слать в облако
- Fallback при отсутствии интернета / исчерпанном бюджете Claude API
- Прогонка агентских пайплайнов на dev (дёшево гонять prompt'ы без счёта)
- "Объясни что делает этот код" / RAG-suммари

**Не подходит:**
- Автономные агенты без ручного review
- Длинные одношотовые генерации (3+ минуты на 2K токенов)
- Latency-sensitive UX (TTF + 10 tok/s стрим — заметная пауза)

## История инфры (что было сломано и как чинили)

1. **Ollama 0.16.1** не знала архитектуру `laguna` (`error loading model architecture: unknown`).
   Лечение: образ → `ollama/ollama:latest` (на момент апдейта 0.23.2). Архитектура laguna появилась в Ollama где-то после 0.16.

2. **Docker Desktop 4.9.1** (май 2022) не пробрасывал GPU через `deploy.resources.reservations.devices` — `nvidia-smi` в контейнере работал, но реальный CUDA runtime hook не активировался → Ollama видела только CPU.
   Лечение: апдейт Docker Desktop до 4.40+ (Engine 29.x, Compose v5+). Auto-updater старой версии не справился, ставили свежий установщик поверх.

3. **PowerShell ConvertTo-Json + Invoke-RestMethod** портит сложные тела с массивами объектов при отправке в Ollama. Workaround: писать JSON в temp-файл, постить через `-InFile`. См. `stress-test.ps1`.

## Полезные команды

```powershell
# Поднять / перезапустить
cd C:\Users\Diamond\Desktop\slovo-llm; docker compose up -d

# Реалтайм-мониторинг GPU из контейнера
docker exec ollama-laguna nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv -l 2

# Бенчмарк (PowerShell, через temp-файл — НЕ через Body, см. п. 3 истории)
& "C:\Users\Diamond\Desktop\slovo-llm\stress-test.ps1"

# Проверить версию Ollama в работающем контейнере
docker exec ollama-laguna ollama --version
```

## Файлы в проекте, на которые опираемся

- `docker-compose.yml` — текущий рабочий конфиг (image: latest, GPU через deploy.resources, env-переменные Ollama)
- `stress-test.ps1` — нагрузочный тест, измеряет prompt_eval/gen rate, мониторит GPU (через `-InFile`, не `-Body`)
- `stress-test-output.md` — последний сэмпл вывода модели, для оценки качества

## 👤 Стиль Димы — ПРИКЛАДНИК (как со мной работать)

Дима = практик/прикладник, **НЕ теоретик**. Как работать:
- **НЕ разводить полемику** («а давай обсудим…») — предлагать и **СРАЗУ делать** дешёвую обратимую пробу. Длинные деференсы по обратимым решениям его раздражают и тормозят.
- **«Нет компетенции» = не стоп, а повод итерировать быстрее** (петля гипотеза→дешёвый тест→урок; понимание приходит ИЗ проб). Breadth-first эмпирика > лобовой экспертный заход.
- **Упёрся в стену → откат → дальше.** Стена = информация, не провал. Берём обратимые ставки — потому откат дёшев.
- **Смотреть на ВСЮ систему** (правила/constraints/побочные механики/среду), не только узкую задачу — здесь его сила, часто решает то что другие не увидели.
- **Failures = сигналы** сужающие поиск, хороший оптимизм (opportunities не risks), **Дима = ЛПР** (decisions actively, не просить approval на каждый шаг).

## 📚 Подпроект school-llm (перенос из ML/C_school_llm, 01.06.2026)

**РЕАЛЬНЫЙ бизнес-проект** (LLM-инференс школьных вопросов, рус.). Перенесён сюда из контеста Yandex ML Cup C. Работаем здесь.
- **`school-llm/START_HERE.md` — ЧИТАТЬ ПЕРВЫМ** (точка входа: порядок чтения + старт-чеклист + первые шаги).
- `school-llm/RESULTS_HISTORY.md` — вся история: версии/баллы (финал C=73), что работает/нет, инсайты, диагноз почему застряли.
- `school-llm/AUTONOMOUS_PLAN.md` — план 73→85+ (Гордеева взяла 85.59 с того же 73).
- `school-llm/{data,anchors,validation,rag_index_gold}/` — hold-out+gold, калибровочные якоря, код инференса, RAG.

**Судья = БЕСПЛАТНАЯ модель по API:** **Laguna (Poolside)** — локально `ollama-laguna` localhost:11434 ($0) ИЛИ Poolside API (`POOLSIDE_API_KEY`). **Кросс-семейство к Qwen-солверам = честно.** ⚠️ laguna шарится с slovo-backend — перед тяжёлым прогоном проверить `~/.claude/AGENT-STATUS.md`.

**Веса (6GB) НЕ перенесены** — референс в `ML/C_school_llm/weights_*` (копировать/симлинкать когда нужно).
