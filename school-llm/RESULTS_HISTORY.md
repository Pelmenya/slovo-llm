# School-LLM — полная история и результаты (перенос из ML/C_school_llm)

> Задача C из Yandex ML Challenge 2026 Long Tour: **эффективный LLM-инференс школьных вопросов** (рус.).
> **Это РЕАЛЬНЫЙ бизнес-проект** (≠ контестные A/B). Контест закрыт 31.05.2026 — продолжаем АВТОНОМНО здесь, в slovo-llm.
> Перенесено 01.06.2026. Источник: `Desktop/ML/C_school_llm/`. План вперёд: `school-llm/AUTONOMOUS_PLAN.md`.

## Условие задачи
- Вход: школьные вопросы (рус., разные предметы: математика, литература, русский, физика, химия, биология, история, обществознание, география, английский).
- Выход: ответы. Оценка — **обученный reward-judge** (correctness + похожесть на gold-эталоны).
- **Серверный envelope:** 1× GPU L4 24GB, 15GB RAM, 8 CPU, без интернета, **15 мин на 4000 вопросов** (≈225ms/вопрос). zip ≤10GB.
- **Финальный результат: C = 73** (Qwen3-8B-AWQ plain SP max384). Топ задачи = **87.71** (Кирпиченко), Гордеева 85.59.

## Природа судьи (доказано)
Reward-model, **чувствительна к КОРРЕКТНОСТИ**, карает уверенно-неправильное; surface-«качество»/длина/формат НЕ компенсируют ошибку. Доказательства: full-SFT 0.6B на gold = 32.3 (НИЖЕ base 0.6B 45.9 — SFT превратил осторожную кроху в уверенного фабрикатора). plain SP (естественное поведение) = лучшее.

## Полная история сабмитов (авторитетные серверные баллы)
| Версия | Конфиг | Вердикт | Балл |
|---|---|---|---|
| v1 | Qwen3-0.6B baseline | OK | 45.9 |
| v2 | Qwen3-4B-AWQ + plain SP + max384 greedy | OK | 66 |
| v3 | 4B-AWQ + markdown SP + max1024 | TL | 0 |
| v8 | 4B-AWQ + LoRA r16 markdown + max384 | TL | 0 |
| v9 | 4B-AWQ + LoRA + max180 | OK | 59.2 |
| v_probe | 4B-AWQ markdown max180 (no LoRA) | OK | 55.6 |
| v_safe | 8B-AWQ plain max100 | OK | 66.3 |
| **v_safe_384** | **8B-AWQ plain SP max384** | OK | **73** ⭐ ФИНАЛ |
| v_lora5_* (384/320/296) | 8B + LoRA v5 (Haiku/Sonnet distilled) | TL | 0 |
| v_safe_384_sampling | 8B plain max384 + temp0.3+min_p | TL | 0 |
| v_fewshot | 8B plain + 4 in-context examples max384 | OK | 70.8 |
| v_fewshot2 | 8B plain + 2 minimal examples | OK | 70.5 |
| v_14b_safe_256 | Qwen3-14B-AWQ plain max256 | RE | 0 |
| v_fewshot_plain_lora5 | 8B + fewshot_plain + LoRA v5 | OK | 57.2 ❌ |
| v_qwen25_14b_safer | Qwen2.5-14B-AWQ plain max200 | OK | 68.7 |
| v_qwen25_14b_max384 | Qwen2.5-14B-AWQ plain max384 | OK | 69.6 |
| v_qwen3_14b_safer_384 | Qwen3-14B-AWQ max384 max_len2048 | TL 19м | 0 |
| v_qwen3_14b_safer_384_v2 | same max_len1536 | TL 19м | 0 |
| v_lora6_compact_384 | 8B+LoRA v6 compact max_len1152 | RE | 0 |
| v_lora6_compact ENVELOPE | same + envelope fix | TL 18м | 0 |
| **30.05+ финальная серия:** | | | |
| 0.6B full-SFT (r64) | full-SFT на сыром gold | OK | **32.3** (ниже base!) |
| plain2 (answer-first SP) | 8B answer-first | OK | 66.2 |
| genrag | 8B routing+selective-RAG | OK | 66.4 |
| T-lite bnb-4bit | T-lite-it (Qwen2.5-7.6B рус) plain | OK | **70.3** |
| Qwen3-14B max150 @0.92 | влез по TL | OK | ~71.6 |
| 8b_nolang | plain БЕЗ «по-русски» (English-фикс) | (RE-context) | ~72.97 |
| v_safe_512 | plain max384→512 | OK | **73** (=идентично, cap редко бьётся) |

## Что РАБОТАЕТ / НЕ работает (карта)
**Работает:** Qwen3-8B-AWQ + plain SP + max384 = **73 (потолок нашего подхода).** Естественное поведение 8B, greedy.

**НЕ работает (всё <73):**
- **LoRA fine-tune** — net-negative. r8 раздувает длину (→TL), r16 ломает reasoning (math/рус), quant-mismatch (bnb-train→AWQ-deploy) ломает. Судья любит дефолт. (`project_lora_finetune_net_negative_c`)
- **full-SFT 0.6B = 32.3** — хуже базы (фабрикатор).
- **14B** — Qwen3-14B TL (decode-bound, full=19мин); Qwen2.5-14B каппится 69.6; max150-fit=71.6. token-starved.
- **T-lite (Qwen2.5-7.6B рус) = 70.3** — знает рус.факты (Хлестаков, морфология) где Qwen3-8B фабрикует, НО Qwen2.5-семья стиль-mismatch с gold → каппится <73.
- **RAG (genrag) = 66.4** — distribution shift от gold навредил (прокси+eyeball обманули).
- **Промпты:** markdown=55, answer-first(plain2)=66.2 (убил CoT→сломал math), fewshot=70.5-70.8, nolang=72.97 (English-фикс опровергнут).
- **spec-decode ЗАКРЫТ** — vLLM 0.11 НЕ поддерживает draft-model; ngram=−15% (вредит bulk-батчу).

## Ключевые инсайты
1. **Судья = correctness + gold-style-similarity** (gold вероятно генерён Qwen3-семьёй → Qwen3-8B максимально совпадает → 73).
2. **Наш 73-кап = knowledge-gap:** Qwen3-8B **фабрикует русскую школу** (литература/русский — путает Гоголя, выдумывает предложения, неверные худ.средства). Math/наука — ок. Галлюцинации = knowledge-gap, НЕ размер/длина.
3. **TL — управляемая ось:** SP-концентрация = регулятор длины; wall = anchor × avg-output-токены. 8B-AWQ 145ток=14мин, 14B 58ток=10мин.
4. **Прокси combined (cos+rouge+exact) = ФИГНЯ для judge** (любит длину/формат, врёт; K=161.8 не переносится). Доверять только: токен-длина + EYEBALL + (был) сервер.

## КОРНЕВОЙ диагноз — почему застряли на 73 (НЕ некомпетентность)
Перепробовали вагон (LoRA r8/16/32, RAG, 14B, T-lite, full-SFT, routing, промпты). Killer:
1. **СЛЕПОТА** — не было надёжного дешёвого прокси судьи → каждую идею проверяли ТОЛЬКО серверной попыткой (вслепую) → вагон идей = вагон сожжённых попыток.
2. **Бюджет попыток** — в начале контеста правило «последний сабмит» → защитные перезаливки = N/2 эксперимента. best-attempt включили 30.05, но бюджет уже выгорел. Утром были на 73 как Гордеева — она дотолкала до 85.59 (был запас), мы нет.

**→ Лекарство (этот проект):** БЕСПЛАТНАЯ локальная петля валидации (LLM-judge) → проверяем КАЖДУЮ идею за $0, отсеиваем мусор локально. Барьер снят: нет лимита попыток, нет смены правил.

## Цель проекта = 73 → 85+
Планка доказана (Гордеева 85.59 с того же 73). Knowledge-gap (рус.школа) = рычаг. Подробный план: `AUTONOMOUS_PLAN.md`.

## Ключевые артефакты (в `Desktop/ML/C_school_llm/`)
- Hold-out: `validation/splits/valid_1000.pkl` (gold внутри), `train_9000.parquet`.
- Веса: `weights_8b_awq` (73), `weights_tlite_bnb4bit` (70.3), `weights_qwen3_14b_awq`.
- Сохранённые ответы valid_1000: `work/*_valid1000.json`, `gen_rag_1000.json`, `lora06b_r64_v1000.json`.
- Инференс: `validation/run_inference.py`, RAG: `run_inference_rag.py`, `work/rag_index_gold/`.
- Memory: `project_c_real_business`, `project_c_path_to_85_postcontest`, `project_lora_finetune_net_negative_c`.
