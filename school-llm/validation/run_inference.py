"""Прогон модели на valid split → predictions.json.

Все варианты v4/v5/v6/v7 запускаются одним и тем же entrypoint с разными флагами,
чтобы сравнение было strictly apples-to-apples (та же модель, тот же seed, тот же
input pickle, разница только в конфиге).

Output: JSON с `config`, `wall_seconds`, `tok_per_s`, `predictions[]`.
"""
import argparse
import json
import os
import pickle
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


SYSTEM_PROMPTS = {
    "plain": (
        "Ты — школьный репетитор. Отвечай по-русски, кратко и по делу, "
        "без вступлений и markdown-разметки. Если задача с уравнением — пиши шаги. "
        "Если требуется длинный ответ (сочинение, объяснение) — отвечай развёрнуто."
    ),
    # v8plain — ТОЧНАЯ копия train_lora_v8plain.py SYSTEM_PROMPT (train==deploy, urok v5).
    "v8plain": (
        "Ты — школьный репетитор. Отвечай по-русски кратко и точно, сразу по делу — "
        "без вступлений, без повторения вопроса, без markdown, без жирного и заголовков.\n"
        "Дай прямой ответ и только необходимое пояснение, обычно 1–4 предложения или "
        "короткий список.\n"
        "В литературе, истории, биологии, географии указывай конкретные имена, даты и "
        "термины. Не выдумывай факты, имена, цитаты и даты — если не уверен, не добавляй.\n"
        "Не задавай уточняющих вопросов."
    ),
    # plain_nolang — proven plain БЕЗ "по-русски" (Дима: судья штрафует англ-вопросы отвеченные по-русски)
    "plain_nolang": (
        "Ты — школьный репетитор. Отвечай кратко и по делу, "
        "без вступлений и markdown-разметки. Если задача с уравнением — пиши шаги. "
        "Если требуется длинный ответ (сочинение, объяснение) — отвечай развёрнуто."
    ),
    # plain2 — минимальный diff от proven plain (73): answer-first + явный итог. TL-safe (без удлинения).
    "plain2": (
        "Ты — школьный репетитор. Отвечай по-русски, кратко и по делу, "
        "без вступлений и markdown-разметки. Сначала дай прямой ответ, затем кратко поясни. "
        "Если задача с уравнением — пиши шаги и итог. "
        "Если требуется длинный ответ (сочинение, объяснение) — отвечай развёрнуто."
    ),
    # plain3 — answer-first (как plain2) НО без удлинения: пояснение 1-2 предл, без лазейки "развёрнуто". TL-safe.
    "plain3": (
        "Ты — школьный репетитор. Отвечай по-русски кратко и по делу, "
        "без вступлений и markdown-разметки. Сначала дай прямой точный ответ, "
        "затем короткое пояснение в 1–2 предложения. "
        "Если задача с уравнением — пиши шаги и итог."
    ),
    # plain4 — answer-first sweet spot: ~120 ток (между plain3=85 и plain2=138). TL-safe + не под-ответ.
    "plain4": (
        "Ты — школьный репетитор. Отвечай по-русски точно и по делу, "
        "без вступлений и markdown-разметки. Сначала дай прямой ответ, "
        "затем кратко поясни (2–3 предложения). "
        "Если задача с уравнением — пиши шаги и итог."
    ),
    # v8plain2 — ТОЧНАЯ копия train_lora_v8plain.py SYSTEM_PROMPTS["v8plain2"] (придавленный)
    "v8plain2": (
        "Ты — школьный репетитор. Отвечай концентрированно: только суть и "
        "ответ, 1–3 коротких предложения или короткий список.\n"
        "Без markdown и жирного."
    ),
    "markdown": (
        "Ты — школьный репетитор. Отвечай по-русски, по делу.\n"
        "Используй Markdown-разметку как в учебниках:\n"
        "- **жирным** выделяй ключевые термины, ответ, важные факты;\n"
        "- математические формулы — в LaTeX-нотации `$...$` для строки и `$$...$$` для выкладок;\n"
        "- шаги решения — нумерованным списком;\n"
        "- для химических реакций используй стандартную нотацию (H₂O, CO₂, →)."
    ),
    "compact": (
        "Ты — школьный репетитор. Отвечай по-русски, СЖАТО и ПО ДЕЛУ.\n"
        "Используй Markdown ТОЛЬКО где нужно:\n"
        "- **жирным** только ключевой термин или финальный ответ;\n"
        "- математика — `$LaTeX$` для формул;\n"
        "- шаги — нумерованным списком (максимум 5 пунктов).\n"
        "Без вступлений, без \"итак, давайте разберём\" — сразу к ответу."
    ),
    # compact_cap — GPT-5.5 recommendation для max=128 deploy:
    # cap-aware: "сначала ответ, потом объяснение" → modal frontloads result
    # → не теряет суть если cap binding
    "compact_cap": (
        "Ты — школьный репетитор. Отвечай по-русски точно, понятно и "
        "компактно.\n\n"
        "Сначала дай главный ответ, затем краткое объяснение.\n"
        "Если есть вычисления — покажи только основные шаги и итог.\n"
        "Если нужны формулы, термины или короткий список — используй их.\n"
        "Не пиши длинные вступления, не повторяй вопрос, не задавай "
        "уточняющие вопросы.\n"
        "Старайся уложиться в 3–7 предложений."
    ),
    # compact SP — GPT-5.5 recommendation для LoRA v6 deploy:
    # НЕ запрещает markdown (alignment с rich train) но просит кратко
    # Target: avg output 120-135 tokens / 450-550 chars
    "compact": (
        "Ты — школьный репетитор. Отвечай по-русски точно, понятно и "
        "достаточно кратко.\n\n"
        "Дай готовый ответ без лишних вступлений.\n"
        "Если задача с вычислениями — покажи основные шаги и итог.\n"
        "Если нужны формулы, термины или короткий список — используй их.\n"
        "В гуманитарных предметах называй конкретные факты, авторов, героев "
        "и термины.\n"
        "Не пиши длинные рассуждения, не повторяй вопрос, не задавай "
        "уточняющие вопросы."
    ),
    # rich SP — matches train_lora_v6 SP exactly (allows markdown, aligned с judge preferences)
    # Использовать с LoRA v6+ adapter trained on rich SP
    "rich": (
        "Ты — школьный репетитор. Отвечай по-русски точно и развёрнуто.\n"
        "- Структурируй ответ для ясности: **жирным** выделяй ключевые "
        "термины и финальный ответ; для формул используй LaTeX в долларах "
        "($...$ для строки, $$...$$ для выкладок); шаги нумеруй списком.\n"
        "- Если задача с вычислениями — пиши последовательно: что считаем, "
        "формула, подстановка, ответ.\n"
        "- В литературе, истории, биологии — указывай конкретных авторов, "
        "героев, термины. Не выдумывай имена и факты.\n"
        "- Развёрнутые ответы — последовательно: главная мысль, аргументы, "
        "вывод.\n"
        "- Не задавай уточняющие вопросы, отвечай по существу."
    ),
    # v7compact — ТОЧНАЯ копия train_lora_v7.py SYSTEM_PROMPT (train==deploy, v5 lesson).
    # compact-rich: сохраняет формат, просит компактность. Без "сначала ответ".
    "v7compact": (
        "Ты — школьный репетитор. Отвечай по-русски: точно, структурно и компактно.\n"
        "- **Жирным** выделяй ключевые термины и финальный ответ; формулы — в LaTeX "
        "($...$ для строки, $$...$$ для выкладок); шаги решения — нумерованным списком.\n"
        "- В литературе, истории, биологии указывай конкретных авторов, героев, "
        "термины. Не выдумывай имена и факты.\n"
        "- Не растягивай ответ: обычно 3–7 предложений или короткий список. "
        "Без вступлений и воды.\n"
        "- Не задавай уточняющие вопросы, отвечай по существу."
    ),
    # v7terse — терсный SP: rich-минимум (bold/LaTeX ок), но жёстко короткий + ПОЛНЫЙ ответ.
    # Гипотеза: длину гонит SP, не LoRA. Цель ~150 ток complete (без обрезки).
    "v7terse": (
        "Ты — школьный репетитор. Отвечай по-русски кратко и по существу — 2–5 предложений.\n"
        "Сразу давай ответ, без вступлений и воды. **Главный ответ** можно выделить жирным, "
        "формулы — в LaTeX ($...$). Не пиши длинных списков, заголовков и развёрнутых разборов.\n"
        "Заверши ответ законченной мыслью."
    ),
    # v8min — минимальный plain (GPT: LoRA несёт стиль в весах, SP только давит длину/формат)
    "v8min": (
        "Дай краткий точный ответ по-русски, без markdown и лишних пояснений. "
        "Обычно 1–3 предложения."
    ),
    # v7lean — таргетный: компактно, лёгкое форматирование для технических (не мандат),
    # content-гайд для слабых литература/русский (где формат не помогает per measurement).
    "v7lean": (
        "Ты — школьный репетитор. Отвечай по-русски точно, по делу и компактно, без вступлений.\n"
        "- Математика, физика, химия: пиши шаги и итог; формулы — в LaTeX ($...$); "
        "ключевой ответ выделяй **жирным**.\n"
        "- Литература: называй автора, произведение и героев; разбирай образы, темы и "
        "авторскую позицию, опирайся на текст.\n"
        "- Русский язык: указывай точные термины (часть речи, падеж, разбор) и правило.\n"
        "- Не выдумывай имена, даты и факты. Обычно хватает 3–6 предложений или "
        "короткого списка."
    ),
    # v7af — Дима 29.05: answer-first, terse, anti-«мини-сочинение», БЕЗ жирного/markdown.
    # Чинит killer v7lean (раздувание гуманитарки + разметка). Цель: ≤145 ток на longest.
    "v7af": (
        "Ты — школьный репетитор. Отвечай по-русски кратко и точно, без вступлений.\n\n"
        "Дай сразу ответ. Затем добавь 1–3 коротких шага или аргумента, если без них "
        "ответ будет непонятен.\n"
        "В вычислениях: действие → подстановка → итог.\n"
        "В гуманитарных вопросах: ключевой факт, краткое объяснение, вывод.\n"
        "Не повторяй вопрос, не пиши мини-сочинение, не выдумывай факты.\n"
        "Без markdown-заголовков и без жирного текста."
    ),
    # v7conc — Дима 29.05: максимально концентрированный. Цель — короткие выходы,
    # чтобы даже 14B влез по TL (14B качество + brevity = TL-safe).
    "v7conc": (
        "Ты — репетитор. От тебя нужен строго концентрированный и точный ответ.\n"
        "Дай только суть, прямо отвечающую на вопрос: без вступлений, без повторения "
        "вопроса, без воды и рассуждений вслух.\n"
        "В вычислениях — только действия и итог.\n"
        "Без markdown и жирного текста. Не выдумывай факты."
    ),
    # v7conc2 — Дима 29.05: ослабленный v7conc (~85 ток target, используем 5-мин запас 14B).
    # 14B@56ток=10мин → ~85ток≈14мин safe. Цель: больше содержания → bet >73.
    "v7conc2": (
        "Ты — репетитор. Дай точный, концентрированный ответ.\n"
        "Сначала — суть/итог, затем краткое пояснение (1–3 предложения) "
        "или основные шаги для вычислений.\n"
        "Без вступлений, без повторения вопроса, без воды, без markdown и жирного.\n"
        "Не выдумывай факты."
    ),
    "fewshot_plain": (
        "Ты — школьный репетитор. Отвечай по-русски точно и по существу.\n"
        "- Без markdown-разметки, без bold и заголовков.\n"
        "- Если задача с вычислениями — пиши шаги: что считаем, подстановка, ответ.\n"
        "- В литературе, истории, биологии — указывай конкретных авторов, героев, "
        "термины. Не выдумывай имена и факты.\n"
        "- Если развёрнутый ответ — пиши последовательно: главная мысль, "
        "аргументы, вывод.\n"
        "- Не задавай уточняющие вопросы, отвечай по существу.\n\n"
        "Примеры хороших ответов:\n\n"
        "Вопрос: Что такое антитеза в литературе?\n"
        "Ответ: Антитеза — стилистический приём, при котором в тексте "
        "противопоставляются два понятия, образа или явления. Например, у "
        "Лермонтова: волна и камень, стихи и проза, лёд и пламень. Антитеза "
        "усиливает контраст и делает речь выразительнее.\n\n"
        "Вопрос: В каком падеже стоит слово школу в предложении Я иду в школу?\n"
        "Ответ: Слово школу стоит в винительном падеже. Вопрос: иду куда? — в "
        "школу. Винительный падеж неодушевлённых существительных отвечает на "
        "вопросы кого? что?.\n\n"
        "Вопрос: Реши: 12 умножить на 8 плюс 35\n"
        "Ответ: Сначала умножение: 12 на 8 равно 96. Затем сложение: 96 плюс "
        "35 равно 131. Ответ: 131.\n\n"
        "А теперь отвечай на следующий вопрос в том же стиле:"
    ),
    "fewshot": (
        "Ты — школьный репетитор. Отвечай по-русски, кратко и по делу, "
        "без вступлений и markdown-разметки. Если задача с уравнением — пиши шаги. "
        "Если требуется длинный ответ (сочинение, объяснение) — отвечай развёрнуто.\n\n"
        "Примеры хороших ответов:\n\n"
        "Вопрос: расставьте коэффициенты в уравнении химической реакции AlCl3 + NaOH\n"
        "Ответ: $$\\text{AlCl}_3 + 3\\,\\text{NaOH} \\to \\text{Al(OH)}_3 \\downarrow + 3\\,\\text{NaCl}$$\n\n"
        "Уравнивание: Al: 1=1, Cl: 3=3, Na: 3=3, O и H: 3=3.\n\n"
        "**Коэффициенты: 1, 3, 1, 3**. Гидроксид алюминия выпадает в осадок.\n\n"
        "Вопрос: реши уравнение: (x − 4 5/17) + 1 8/17 = 6 12/17\n"
        "Ответ: Переведём в неправильные дроби: $4\\tfrac{5}{17} = \\tfrac{73}{17}$, "
        "$1\\tfrac{8}{17} = \\tfrac{25}{17}$, $6\\tfrac{12}{17} = \\tfrac{114}{17}$.\n\n"
        "Уравнение: $x - \\tfrac{73}{17} + \\tfrac{25}{17} = \\tfrac{114}{17}$\n\n"
        "$x - \\tfrac{48}{17} = \\tfrac{114}{17}$ → $x = \\tfrac{162}{17}$.\n\n"
        "**Ответ: $x = 9\\tfrac{9}{17}$**\n\n"
        "Вопрос: Приведи 5 примеров использования глагола to be в Present Simple.\n"
        "Ответ: Глагол **to be** в Present Simple: *am* (I), *is* (he/she/it), *are* (you/we/they).\n\n"
        "1. I am a student. (Я студент.)\n"
        "2. She is from Canada. (Она из Канады.)\n"
        "3. They are friends. (Они друзья.)\n"
        "4. It is cold today. (Сегодня холодно.)\n"
        "5. We are at home. (Мы дома.)\n\n"
        "**Используется для описания состояния, происхождения, местонахождения и характеристик.**\n\n"
        "Вопрос: что такое фотосинтез кратко\n"
        "Ответ: Процесс синтеза органических веществ (глюкозы) из углекислого газа и воды "
        "в зелёных растениях под действием света:\n\n"
        "$$6CO_2 + 6H_2O \\xrightarrow{\\text{свет, хлорофилл}} C_6H_{12}O_6 + 6O_2$$\n\n"
        "**Значение:** образование органики и выделение кислорода — основа жизни на Земле.\n\n"
        "А теперь отвечай на следующий вопрос в том же стиле:"
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="путь к weights/ с safetensors")
    parser.add_argument("--input", required=True, help="valid pkl с [{rid, question, reference}]")
    parser.add_argument("--output", required=True, help="path для predictions.json")
    parser.add_argument("--system-prompt", choices=list(SYSTEM_PROMPTS), default="plain")
    parser.add_argument("--route", action="store_true",
                        help="per-question SP routing через route_sp.py (игнорит --system-prompt)")
    parser.add_argument("--quantization", default="awq", help="awq | none")
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--rep-penalty", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--gpu-mem-util", type=float, default=0.85)
    parser.add_argument("--max-num-seqs", type=int, default=0, help="0 = vLLM default; >0 ограничить batch (14B на 16GB)")
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--no-prefix-caching", action="store_true")
    parser.add_argument("--pre-sort", action="store_true",
                        help="отсортировать prompts по длине desc для vLLM continuous batching")
    parser.add_argument("--variant-tag", default="unknown",
                        help="строковый тэг варианта, попадает в output для трекинга")
    parser.add_argument("--lora-dir", default=None,
                        help="опционально: путь к LoRA адаптеру (для v8+ inference)")
    parser.add_argument("--max-lora-rank", type=int, default=16)
    parser.add_argument("--adaptive-tokens", action="store_true",
                        help="per-query max_tokens по classifier (short=80, medium=200, long=384)")
    parser.add_argument("--enable-thinking", action="store_true",
                        help="включить Qwen3 thinking mode (<think>...</think>answer); "
                             "answer-only извлекается из output после закрывающего тега")
    args = parser.parse_args()

    sp_text = SYSTEM_PROMPTS[args.system_prompt]

    with open(args.input, "rb") as f:
        rows = pickle.load(f)
    print(f"loaded {len(rows)} valid rows from {args.input}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)

    llm_kwargs = dict(
        model=args.model_dir,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        enable_prefix_caching=not args.no_prefix_caching,
        enable_chunked_prefill=args.enable_chunked_prefill,
        tokenizer_mode="auto",
        seed=0,
    )
    if args.quantization and args.quantization.lower() != "none":
        llm_kwargs["quantization"] = args.quantization
        if args.quantization.lower() == "bitsandbytes":
            llm_kwargs["load_format"] = "bitsandbytes"

    if args.lora_dir:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = args.max_lora_rank
        llm_kwargs["max_loras"] = 1

    print(f"loading vLLM with {llm_kwargs}")
    t_load_start = time.time()
    llm = LLM(**llm_kwargs)
    load_seconds = time.time() - t_load_start
    print(f"vLLM loaded in {load_seconds:.1f}s")

    lora_request = None
    if args.lora_dir:
        from vllm.lora.request import LoRARequest  # noqa: WPS433
        lora_request = LoRARequest("v8_lora", 1, args.lora_dir)
        print(f"using LoRA adapter from {args.lora_dir}")

    if args.route:
        from route_sp import route_sp, classify  # noqa: WPS433
        cats = {}
        for r in rows:
            cats[classify(r["question"])] = cats.get(classify(r["question"]), 0) + 1
        print(f"route categories: {cats}")
        sp_per_row = [route_sp(r["question"]) for r in rows]
    else:
        sp_per_row = [sp_text] * len(rows)

    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": sp_per_row[i]},
                {"role": "user", "content": row["question"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        for i, row in enumerate(rows)
    ]

    if args.pre_sort:
        order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
        ordered_prompts = [prompts[i] for i in order]
    else:
        order = list(range(len(prompts)))
        ordered_prompts = prompts

    def _make_sampling(max_t: int) -> SamplingParams:
        return SamplingParams(
            temperature=args.temperature,
            max_tokens=max_t,
            top_k=args.top_k,
            min_p=args.min_p,
            repetition_penalty=args.rep_penalty,
            presence_penalty=args.presence_penalty,
        )

    if args.adaptive_tokens:
        from classify_query import classify_query, CATEGORY_MAX_TOKENS  # noqa: WPS433
        sampling = [_make_sampling(CATEGORY_MAX_TOKENS[classify_query(rows[i]["question"])])
                    for i in order]
        cat_counts = {"short": 0, "medium": 0, "long": 0}
        for i in order:
            cat_counts[classify_query(rows[i]["question"])] += 1
        print(f"adaptive max_tokens distribution: {cat_counts}")
    else:
        sampling = _make_sampling(args.max_tokens)

    t0 = time.time()
    gen_kwargs = {"sampling_params": sampling}
    if lora_request is not None:
        gen_kwargs["lora_request"] = lora_request
    outputs = llm.generate(ordered_prompts, **gen_kwargs)
    wall = time.time() - t0

    result = [None] * len(rows)
    total_out_tokens = 0
    cot_completed = 0
    cot_cutoff = 0
    raw_total_chars = 0
    answer_total_chars = 0
    for ord_idx, out in zip(order, outputs):
        raw = out.outputs[0].text
        total_out_tokens += len(out.outputs[0].token_ids)
        raw_total_chars += len(raw)
        if args.enable_thinking:
            close_tag = "</think>"
            idx = raw.find(close_tag)
            if idx >= 0:
                ans = raw[idx + len(close_tag):].strip()
                cot_completed += 1
            else:
                ans = raw.strip()
                cot_cutoff += 1
        else:
            ans = raw.strip()
        answer_total_chars += len(ans)
        result[ord_idx] = {
            "rid": rows[ord_idx]["rid"],
            "question": rows[ord_idx]["question"],
            "reference": rows[ord_idx]["reference"],
            "answer": ans,
        }

    payload = {
        "variant_tag": args.variant_tag,
        "config": {k: v for k, v in vars(args).items() if k not in ("input", "output", "model_dir")},
        "n": len(rows),
        "vllm_load_seconds": load_seconds,
        "wall_seconds": wall,
        "total_output_tokens": total_out_tokens,
        "tok_per_s": total_out_tokens / wall if wall > 0 else 0.0,
        "raw_avg_chars": raw_total_chars / len(rows) if rows else 0.0,
        "answer_avg_chars": answer_total_chars / len(rows) if rows else 0.0,
        "cot_completed": cot_completed,
        "cot_cutoff": cot_cutoff,
        "predictions": result,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n[{args.variant_tag}] wall={wall:.1f}s  tokens={total_out_tokens}  tok/s={total_out_tokens / wall:.1f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
