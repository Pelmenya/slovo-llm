# START HERE — school-llm (читать ПЕРВЫМ в новой сессии)

> Привет, будущий я. Это **school-llm** — РЕАЛЬНЫЙ бизнес-проект Димы (LLM-инференс школьных вопросов, рус.), перенесён из контеста Yandex ML Cup C. Работаем ЗДЕСЬ, в slovo-llm.

## 📖 Порядок чтения (5 мин → полный контекст)
1. **`RESULTS_HISTORY.md`** — что уже делали: все версии/баллы (финал C=**73**), карта работает/нет, инсайты, диагноз почему застряли.
2. **`AUTONOMOUS_PLAN.md`** — план вперёд: 73 → **85+** (Гордеева взяла 85.59 с того же 73). Судья, калибровка, рычаги.

## 🎯 Цель
Пробить **73 → 85+**. Рычаг = **knowledge-gap**: Qwen3-8B фабрикует русскую школу (литература/русский), math/наука ок. Судья correctness-sensitive → нужна модель/RAG что НЕ фабрикует факты.

## ✅ СТАТУС (01.06.2026): судья ВАЛИДЕН, петля за $0 готова
Судья = **`poolside/laguna-m.1`** cloud API (НЕ локальная ollama — xs.2 пропала из volume, и модели на GPU не держим). Калибровка пройдена (Spearman≈0.9, концы шкалы точны, ловит knowledge-gap). Детали: memory `project_school_llm_judge_setup` + `RESULTS_HISTORY.md` §«Сессия 01.06.2026». Рычаги к 85 ОТЛОЖЕНЫ.

**Как дёрнуть судью (грабли — см. memory):** Poolside `https://inference.poolside.ai/v1`, ключ `POOLSIDE_API_KEY` в `.env`, **только через прокси** `http://host.docker.internal:10810` (РФ-блок), питон В контейнере `ml-cup-b-baseline:local`. Прогон: `docker run --rm -e POOLSIDE_API_KEY=$key -e HTTPS_PROXY=http://host.docker.internal:10810 -v ...\school-llm:/work ml-cup-b-baseline:local python /work/validation/judge_poolside.py --n 60 --workers 16`.

## ⚙️ Старт сессии (чеклист) — устарел, см. СТАТУС выше
1. ~~Поднять локальную laguna~~ → судья теперь cloud API (выше). ⚠️ перед тяжёлым прогоном глянуть `~/.claude/AGENT-STATUS.md`.
2. **Веса solver'ов** (для перегенерации якорей) — референс в `C:\Users\Diamond\Desktop\ML\C_school_llm\weights_*` (`weights_8b_awq`=73, `weights_tlite_bnb4bit`=70.3). Копировать/симлинкнуть когда нужно. **НЕ в git** (6GB).
3. **Docker-only прогон** моделей (не локальный python).

## ▶️ Первые шаги (из AUTONOMOUS_PLAN §5)
1. **Спросить Диму scoping:** (a) use-case (продукт/ученики?); (b) deployment-envelope (дешёвый-GPU+latency или мягче?); (c) что значит «хороший ответ» (correctness/педагогика/формат?).
2. Поднять судью laguna → тест-вызов: `вопрос + gold + наш_ответ → correctness 0-100`.
3. **Калибровка (make-or-break):** прогнать судью на 6 якорях (`anchors/` + gold из `data/valid_1000.pkl`), проверить РАНЖИРОВАНИЕ **73 > 70.3 > 68.4 > 66.4 > 66.2 > 32.3**. Сойдётся → судья валиден.
4. Валиден → крутить рычаги к 85 (русская модель / RAG / 14B) **бесплатно** на судье.

## 👤 Стиль Димы (КРИТИЧНО)
**Прикладник, НЕ теоретик.** НЕ разводить полемику — предлагать и СРАЗУ делать дешёвую обратимую пробу. «Нет компетенции» = не стоп, а повод итерировать. Смотреть на всю систему. Дима = ЛПР. Полное: slovo-llm `CLAUDE.md` + `~/.claude/CLAUDE.md`.

## 🧠 Главные уроки (не повторять грабли)
- **НЕ объявлять «потолок» по своему потолку** — Гордеева взяла 85, путь есть. Копать дальше.
- Корень прошлого провала = **слепота** (не было дешёвого judge → каждая идея вслепую жгла попытку). Здесь судья бесплатный → проверяем ВСЁ за $0.
- Цель = **РЕАЛЬНОЕ качество** (correctness+полезность), НЕ контест-gold-similarity.
- Прокси cos+rouge+exact = врёт. LLM-judge (Laguna) + eyeball + per-subject = доверять.

## 📁 Что где
- `data/` — valid_1000.pkl (вопрос+gold), train_9000.parquet
- `anchors/` — 5 наборов ответов с известным серверным баллом (для калибровки)
- `validation/` — run_inference.py, run_inference_rag.py, classify_subject.py, score_predictions.py
- `rag_index_gold/` — gold RAG-индекс
- Веса — НЕ здесь (референс ML/C_school_llm)
