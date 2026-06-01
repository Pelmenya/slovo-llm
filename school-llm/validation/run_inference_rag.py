"""Inference c RAG retrieval: top-K похожих (Q, A) пар → in-context examples.

Подход:
1. Загрузить encoder (sentence-transformers) + index (questions.npy + pairs.json)
2. Batch-encode все queries (один прогон encoder)
3. Cosine top-K (numpy dot, float16 normalized)
4. Сформировать prompt = base SP + K examples + user question
5. vLLM generate

Output: predictions.json как у run_inference.py + поле rag_top_qids в каждом entry.
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


BASE_SP = (
    "Ты — школьный репетитор. Отвечай по-русски, кратко и по делу, "
    "без вступлений и markdown-разметки. Если задача с уравнением — пиши шаги. "
    "Если требуется длинный ответ (сочинение, объяснение) — отвечай развёрнуто."
)


def format_prompt_with_examples(question: str, examples: list[dict]) -> tuple[str, str]:
    """Return (system_prompt_with_examples, user_question).

    Examples encoded в system prompt чтобы prefix-cache мог переиспользовать
    общую часть base SP (если retrieval даст совпадение — но обычно не даст,
    examples отличаются). Per-query prefill полный.
    """
    if not examples:
        return BASE_SP, question
    parts = [BASE_SP,
             "\nПохожие проверенные примеры из базы. Используй их ТОЛЬКО если они прямо "
             "относятся к вопросу. Не переноси имена, даты, произведения, персонажей и "
             "правила из нерелевантных примеров. Если не подходят — игнорируй.\n"]
    for ex in examples:
        a = str(ex["a"])[:300]
        if " " in a:
            a = a.rsplit(" ", 1)[0]  # не рвать слово
        parts.append(f"\nВопрос: {ex['q']}\nОтвет: {a}\n")
    parts.append("\nОтвечай на вопрос кратко, точно и по делу. Не выдумывай факты:")
    return "".join(parts), question


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input", required=True, help="valid pkl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rag-index-dir", required=True,
                        help="dir с questions.npy + pairs.json")
    parser.add_argument("--rag-encoder-dir", required=True,
                        help="dir со SentenceTransformer")
    parser.add_argument("--rag-k", type=int, default=2,
                        help="сколько top examples передавать в prompt")
    parser.add_argument("--quantization", default="awq")
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-mem-util", type=float, default=0.85)
    parser.add_argument("--variant-tag", default="rag")
    args = parser.parse_args()

    print(f"loading RAG index from {args.rag_index_dir}…")
    index_q = np.load(os.path.join(args.rag_index_dir, "questions.npy"))
    with open(os.path.join(args.rag_index_dir, "pairs.json"), "r", encoding="utf-8") as f:
        pairs = json.load(f)
    print(f"index: {index_q.shape} dtype={index_q.dtype}, {len(pairs)} pairs")
    # cast к float32 для cosine точности (но index хранится в float16)
    index_q_f32 = index_q.astype(np.float32)

    print(f"loading encoder from {args.rag_encoder_dir}…")
    t_enc_load = time.time()
    encoder = SentenceTransformer(args.rag_encoder_dir, device="cuda")
    print(f"encoder loaded in {time.time() - t_enc_load:.1f}s")

    with open(args.input, "rb") as f:
        rows = pickle.load(f)
    print(f"loaded {len(rows)} rows from {args.input}")

    print("encoding queries…")
    t_enc = time.time()
    queries = [r["question"] for r in rows]
    q_emb = encoder.encode(queries, batch_size=64, show_progress_bar=False,
                           convert_to_numpy=True, normalize_embeddings=True)
    enc_time = time.time() - t_enc
    print(f"encoded {len(queries)} queries in {enc_time:.1f}s "
          f"({len(queries) / enc_time:.0f} q/s)")

    print(f"retrieval top-{args.rag_k}…")
    t_ret = time.time()
    # cosine = dot product так как оба нормализованы
    sims = q_emb.astype(np.float32) @ index_q_f32.T   # (N_query, N_index)
    top_idx = np.argsort(-sims, axis=1)[:, : args.rag_k]
    ret_time = time.time() - t_ret
    print(f"retrieved top-{args.rag_k} in {ret_time:.1f}s")

    print("freeing encoder GPU memory…")
    del encoder
    import torch
    torch.cuda.empty_cache()

    print(f"loading vLLM ({args.model_dir})…")
    t_vllm = time.time()
    llm = LLM(
        model=args.model_dir,
        quantization=args.quantization,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        enable_prefix_caching=True,
        tokenizer_mode="auto",
        seed=0,
    )
    vllm_time = time.time() - t_vllm
    print(f"vLLM loaded in {vllm_time:.1f}s")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)

    prompts = []
    rag_topqids = []
    for i, row in enumerate(rows):
        idxs = top_idx[i].tolist()
        examples = [pairs[j] for j in idxs]
        sp_with_ex, user_q = format_prompt_with_examples(row["question"], examples)
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": sp_with_ex},
                {"role": "user", "content": user_q},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(prompt)
        rag_topqids.append(idxs)

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        top_k=-1,
    )

    print(f"running vLLM generate on {len(prompts)} prompts…")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params=sampling)
    wall = time.time() - t0

    total_out_tokens = 0
    result = []
    for row, out, top in zip(rows, outputs, rag_topqids):
        ans = out.outputs[0].text.strip()
        total_out_tokens += len(out.outputs[0].token_ids)
        result.append({
            "rid": row["rid"],
            "question": row["question"],
            "reference": row.get("reference", ""),
            "answer": ans,
            "rag_top_qids": top,
        })

    payload = {
        "variant_tag": args.variant_tag,
        "config": {k: v for k, v in vars(args).items() if k not in ("input", "output", "model_dir")},
        "n": len(rows),
        "vllm_load_seconds": vllm_time,
        "encode_seconds": enc_time,
        "retrieval_seconds": ret_time,
        "wall_seconds": wall,
        "total_output_tokens": total_out_tokens,
        "tok_per_s": total_out_tokens / wall if wall > 0 else 0.0,
        "predictions": result,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n[{args.variant_tag}] wall={wall:.1f}s  encode={enc_time:.1f}s  "
          f"tok/s={total_out_tokens / wall:.1f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
