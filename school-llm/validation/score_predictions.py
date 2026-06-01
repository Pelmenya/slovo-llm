"""Local scoring of predictions vs reference.

Input: predictions.json from run_inference.py (or run_inference_rag.py).
       должен иметь поля [{rid, question, reference, answer}, ...]

Output:
- scores.json: {overall, per_subject, per_metric, per_question}
- report.md: human-readable summary

Метрики (industry-standard combo для open-ended):
1. cosine_semantic — sentence-transformers paraphrase-multilingual-MiniLM-L12-v2,
   cosine between answer and reference embeddings. Robust to paraphrase.
2. rouge_l — longest common subsequence over tokens. Strong для keyword coverage.
3. exact_match — case-insensitive substring (reference в answer ИЛИ answer в reference) —
   strong сигнал для коротких math/factual ответов.
4. combined — weighted: 0.5*cosine + 0.3*rouge_l + 0.2*exact_match

Per-subject breakdown через validation.classify_subject.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from classify_subject import classify


ENC_PATH = "/root/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/e8f8c211226b894fcb81acc59f3b34ba3efd5f42"


def _tokenize(text: str) -> list[str]:
    """Простая токенизация: lowercase split по non-letters, drop short."""
    import re
    return [t for t in re.split(r"[^а-яёa-z0-9]+", text.lower()) if len(t) >= 2]


def rouge_l_f1(ref_tokens: list[str], hyp_tokens: list[str]) -> float:
    """ROUGE-L F1 score: LCS based."""
    if not ref_tokens or not hyp_tokens:
        return 0.0
    m, n = len(ref_tokens), len(hyp_tokens)
    # dp[i][j] = LCS длина первых i ref tokens и j hyp tokens
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    prec = lcs / n
    rec = lcs / m
    return 2 * prec * rec / (prec + rec)


def exact_match(ref: str, hyp: str) -> float:
    """1.0 если ref ⊆ hyp ИЛИ hyp ⊆ ref (lowercase, stripped); 0.0 иначе.

    Для коротких math ответов вроде "956" — даёт сигнал даже если hyp многословный.
    """
    r = ref.lower().strip()
    h = hyp.lower().strip()
    if not r or not h:
        return 0.0
    if r in h or h in r:
        return 1.0
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="path к predictions.json")
    parser.add_argument("--output-dir", default="work/scores",
                        help="куда писать scores.json + report.md")
    parser.add_argument("--encoder", default=ENC_PATH)
    parser.add_argument("--tag", default="", help="строковый тэг для именования output")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or Path(args.predictions).stem

    print(f"loading {args.predictions}…")
    with open(args.predictions, "r", encoding="utf-8") as f:
        payload = json.load(f)
    preds = payload.get("predictions", payload)
    print(f"loaded {len(preds)} predictions")

    print(f"loading encoder {args.encoder}…")
    enc = SentenceTransformer(args.encoder, device="cuda")

    refs = [p["reference"] for p in preds]
    hyps = [p["answer"] for p in preds]

    print("encoding references + hypotheses…")
    t0 = time.time()
    ref_emb = enc.encode(refs, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    hyp_emb = enc.encode(hyps, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    print(f"encoded in {time.time() - t0:.1f}s")

    cosine = (ref_emb * hyp_emb).sum(axis=1).astype(np.float32)

    results = []
    for p, c in zip(preds, cosine):
        ref_toks = _tokenize(p["reference"])
        hyp_toks = _tokenize(p["answer"])
        r = rouge_l_f1(ref_toks, hyp_toks)
        e = exact_match(p["reference"], p["answer"])
        combined = 0.5 * float(c) + 0.3 * r + 0.2 * e
        subj = classify(p["question"])
        results.append({
            "rid": p.get("rid"),
            "subject": subj,
            "cosine": float(c),
            "rouge_l": r,
            "exact_match": e,
            "combined": combined,
        })

    # aggregate overall
    overall = {
        "n": len(results),
        "cosine": float(np.mean([r["cosine"] for r in results])),
        "rouge_l": float(np.mean([r["rouge_l"] for r in results])),
        "exact_match": float(np.mean([r["exact_match"] for r in results])),
        "combined": float(np.mean([r["combined"] for r in results])),
    }

    # per-subject
    per_subj = defaultdict(list)
    for r in results:
        per_subj[r["subject"]].append(r)
    subj_summary = {}
    for subj, rows in per_subj.items():
        subj_summary[subj] = {
            "n": len(rows),
            "cosine": float(np.mean([r["cosine"] for r in rows])),
            "rouge_l": float(np.mean([r["rouge_l"] for r in rows])),
            "exact_match": float(np.mean([r["exact_match"] for r in rows])),
            "combined": float(np.mean([r["combined"] for r in rows])),
        }

    out_scores = {
        "tag": tag,
        "predictions_path": args.predictions,
        "config": payload.get("config", {}),
        "wall_seconds": payload.get("wall_seconds"),
        "overall": overall,
        "per_subject": subj_summary,
        "per_question": results,
    }
    scores_path = out_dir / f"scores_{tag}.json"
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(out_scores, f, ensure_ascii=False, indent=2)

    # report
    lines = []
    lines.append(f"# Scores: {tag}\n")
    if payload.get("wall_seconds"):
        lines.append(f"- wall: {payload['wall_seconds']:.1f}s on {len(results)} questions")
    lines.append(f"- predictions: {args.predictions}\n")
    lines.append("## Overall\n")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for k in ("cosine", "rouge_l", "exact_match", "combined"):
        lines.append(f"| {k} | **{overall[k]:.4f}** |")
    lines.append("")
    lines.append("## Per subject\n")
    lines.append("| subject | n | cosine | rouge_l | exact_match | combined |")
    lines.append("|---|---|---|---|---|---|")
    for subj in sorted(subj_summary, key=lambda s: -subj_summary[s]["combined"]):
        s = subj_summary[subj]
        lines.append(f"| {subj} | {s['n']} | {s['cosine']:.4f} | {s['rouge_l']:.4f} | "
                     f"{s['exact_match']:.4f} | **{s['combined']:.4f}** |")
    report_path = out_dir / f"report_{tag}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n=== {tag} ===")
    print(f"  combined: {overall['combined']:.4f}  "
          f"cosine: {overall['cosine']:.4f}  "
          f"rouge_l: {overall['rouge_l']:.4f}  "
          f"exact: {overall['exact_match']:.4f}")
    print(f"wrote {scores_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
