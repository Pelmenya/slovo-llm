"""LLM-as-a-judge на Poolside Laguna M.1 (225B, reasoning) — БЕСПЛАТНЫЙ судья.

Цель: дешёвая локальная петля валидации. Калибруем судью на якорях с известным
СЕРВЕРНЫМ баллом. Make-or-break тест = РАНЖИРОВАНИЕ: судья должен выстроить
конфиги в том же порядке, что и серверный reward-judge.

Природа серверного судьи (доказано в RESULTS_HISTORY): reward-model,
чувствительна к КОРРЕКТНОСТИ, карает уверенно-неправильное; surface-качество/
длина/формат НЕ компенсируют ошибку. Кодируем это в prompt.

Кросс-семейство: Laguna (Poolside) != Qwen (наши solver'ы) => честный судья,
нет self-preference bias.

Запуск (в контейнере ml-cup-b-baseline, через прокси, школа смонтирована в /work):
  docker run --rm \
    -e POOLSIDE_API_KEY=$key \
    -e HTTPS_PROXY=http://host.docker.internal:10810 \
    -e HTTP_PROXY=http://host.docker.internal:10810 \
    -v C:\\Users\\Diamond\\Desktop\\slovo-llm\\school-llm:/work \
    ml-cup-b-baseline:local \
    python /work/validation/judge_poolside.py --n 60 --workers 10

Вход: anchors/*.json (predictions с полями rid/question/reference/answer).
Выход: work/judge/raw_<run>.jsonl (per-item, кэш для resume),
       work/judge/summary_<run>.json + печать ranking-таблицы.
"""
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from classify_subject import classify

# Якоря: файл -> (короткий tag, СЕРВЕРНЫЙ балл). Источник: RESULTS_HISTORY.md.
ANCHORS = [
    ("8b_nolang_v1000.json",        "8b_nolang",   72.97),
    ("q3_14b_v7conc_valid1000.json","14b_v7conc",  68.40),
    ("gen_rag_1000.json",           "genrag",      66.40),
    ("base_plain2_valid1000.json",  "plain2",      66.20),
    ("lora06b_r64_v1000.json",      "06b_sft",     32.30),
]

API_URL = "https://inference.poolside.ai/v1/chat/completions"
MODEL = "poolside/laguna-m.1"

SYS_PROMPT = (
    "Ты — строгий и точный эксперт-экзаменатор школьных ответов (русская школа). "
    "Тебе дают ВОПРОС, ЭТАЛОННЫЙ ответ (gold, считается верным) и ОТВЕТ УЧЕНИКА. "
    "Оцени ОТВЕТ УЧЕНИКА по шкале 0-100 по КОРРЕКТНОСТИ относительно эталона и вопроса.\n"
    "Принципы оценки (важно):\n"
    "- Главное — фактическая и смысловая ПРАВИЛЬНОСТЬ. Объём, красота, формат, "
    "развёрнутость НЕ повышают балл сами по себе.\n"
    "- Уверенно изложенный НЕВЕРНЫЙ факт карай сильно (хуже, чем честное 'не знаю').\n"
    "- Если ответ по сути верен, но иначе сформулирован/без LaTeX — балл высокий.\n"
    "- Частично верный — средний балл пропорционально доле корректного.\n"
    "- Пустой/мусорный/не по теме — близко к 0.\n"
    "Думай кратко. В САМОМ КОНЦЕ выведи ОТДЕЛЬНОЙ строкой ТОЛЬКО JSON без пояснений: "
    '{"score": <целое 0-100>, "reason": "<очень кратко>"}'
)


def build_user(question: str, reference: str, answer: str) -> str:
    q = (question or "").replace("﻿", "").strip()
    return (
        f"ВОПРОС:\n{q}\n\n"
        f"ЭТАЛОН (верный):\n{reference}\n\n"
        f"ОТВЕТ УЧЕНИКА:\n{answer}\n\n"
        "Оцени корректность ОТВЕТА УЧЕНИКА (0-100)."
    )


_SCORE_RE = re.compile(r'"score"\s*:\s*(\d{1,3})')
_NUM_RE = re.compile(r"\b(\d{1,3})\b")


def parse_score(content: str):
    """Достаём score из content. Возвращаем (score|None, raw_reason)."""
    if not content:
        return None, ""
    # Берём ПОСЛЕДНИЙ JSON-объект со score (модель выводит его в конце).
    matches = list(_SCORE_RE.finditer(content))
    if matches:
        s = int(matches[-1].group(1))
        return max(0, min(100, s)), content.strip()[-300:]
    # Fallback: последнее число 0-100 в тексте.
    nums = [int(m.group(1)) for m in _NUM_RE.finditer(content) if 0 <= int(m.group(1)) <= 100]
    if nums:
        return nums[-1], content.strip()[-300:]
    return None, content.strip()[-300:]


def judge_one(key, question, reference, answer, max_tokens=1500, retries=4):
    body = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": build_user(question, reference, answer)},
        ],
    }
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.post(API_URL, headers=headers, data=json.dumps(body), timeout=180)
            if r.status_code == 200:
                j = r.json()
                ch = j["choices"][0]
                content = ch["message"].get("content") or ""
                score, reason = parse_score(content)
                if score is not None:
                    return {"score": score, "reason": reason,
                            "finish": ch.get("finish_reason"),
                            "usage": j.get("usage", {})}
                last_err = f"parse_fail finish={ch.get('finish_reason')} content={content[:120]!r}"
                # если обрезалось по длине — дать больше токенов
                if ch.get("finish_reason") == "length":
                    body["max_tokens"] = min(int(body["max_tokens"] * 1.6), 4000)
            else:
                last_err = f"HTTP {r.status_code}: {r.text[:120]}"
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(2 * (attempt + 1))  # backoff
    return {"score": None, "error": last_err}


def pick_rids(by_rid_per_anchor, n):
    """Пересечение rid во всех якорях, равномерный stride-сэмпл n штук (детерминированно)."""
    common = None
    for d in by_rid_per_anchor.values():
        s = set(d.keys())
        common = s if common is None else (common & s)
    rids = sorted(common)
    if n >= len(rids):
        return rids
    stride = len(rids) / n
    return [rids[int(i * stride)] for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors-dir", default=str(Path(__file__).parent.parent / "anchors"))
    ap.add_argument("--out-dir", default=str(Path(__file__).parent.parent / "work" / "judge"))
    ap.add_argument("--n", type=int, default=60, help="вопросов на якорь (парный сэмпл)")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--run-tag", default="calib")
    args = ap.parse_args()

    key = os.environ.get("POOLSIDE_API_KEY")
    if not key:
        sys.exit("POOLSIDE_API_KEY не задан в окружении")

    anchors_dir = Path(args.anchors_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Загружаем якоря -> by_rid: {tag: {rid: pred}}
    by_rid = {}
    meta = {}
    for fname, tag, server in ANCHORS:
        path = anchors_dir / fname
        if not path.exists():
            print(f"[skip] нет файла {path}")
            continue
        payload = json.load(open(path, encoding="utf-8"))
        preds = payload.get("predictions", payload)
        by_rid[tag] = {p["rid"]: p for p in preds}
        meta[tag] = {"server": server, "file": fname, "n_preds": len(preds)}
        print(f"[load] {tag:12} server={server:>6}  n={len(preds)}")

    rids = pick_rids(by_rid, args.n)
    print(f"[sample] парный сэмпл {len(rids)} rid на якорь × {len(by_rid)} якорей "
          f"= {len(rids) * len(by_rid)} вызовов судьи\n")

    # Кэш для resume: raw jsonl, ключ (tag, rid).
    raw_path = out_dir / f"raw_{args.run_tag}.jsonl"
    done = set()
    if raw_path.exists():
        for line in open(raw_path, encoding="utf-8"):
            try:
                row = json.loads(line)
                if row.get("score") is not None:
                    done.add((row["tag"], row["rid"]))
            except Exception:  # noqa: BLE001
                pass
        print(f"[resume] уже готово {len(done)} оценок из {raw_path.name}")

    # Список задач
    tasks = []
    for tag, d in by_rid.items():
        for rid in rids:
            if (tag, rid) in done:
                continue
            p = d[rid]
            tasks.append((tag, rid, p["question"], p["reference"], p["answer"]))

    print(f"[run] {len(tasks)} новых вызовов, workers={args.workers}\n")
    t0 = time.time()
    raw_f = open(raw_path, "a", encoding="utf-8")
    results = []
    errors = 0
    completed = 0

    def work(task):
        tag, rid, q, ref, ans = task
        res = judge_one(key, q, ref, ans)
        res.update({"tag": tag, "rid": rid})
        return res

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, t): t for t in tasks}
        for fut in as_completed(futs):
            res = fut.result()
            raw_f.write(json.dumps(res, ensure_ascii=False) + "\n")
            raw_f.flush()
            completed += 1
            if res.get("score") is None:
                errors += 1
                print(f"  [ERR] {res['tag']}#{res['rid']}: {res.get('error')}")
            if completed % 20 == 0:
                el = time.time() - t0
                rate = completed / el
                eta = (len(tasks) - completed) / rate if rate else 0
                print(f"  {completed}/{len(tasks)}  {rate:.2f}/s  ETA {eta/60:.1f}m  errs={errors}")
    raw_f.close()
    print(f"\n[done] {completed} вызовов за {(time.time()-t0)/60:.1f} мин, errs={errors}")

    # Перечитываем ВЕСЬ raw (включая прошлые resume) -> агрегаты
    all_rows = []
    for line in open(raw_path, encoding="utf-8"):
        try:
            all_rows.append(json.loads(line))
        except Exception:  # noqa: BLE001
            pass
    # последняя удачная оценка на (tag,rid)
    best = {}
    for row in all_rows:
        if row.get("score") is not None:
            best[(row["tag"], row["rid"])] = row["score"]

    # subject lookup из любого якоря
    subj_of = {}
    any_anchor = next(iter(by_rid.values()))
    for rid in rids:
        if rid in any_anchor:
            subj_of[rid] = classify(any_anchor[rid]["question"])

    summary = {"run_tag": args.run_tag, "n_per_anchor": len(rids), "model": MODEL,
               "anchors": {}, "per_subject": {}}
    print("\n=== RANKING (судья vs сервер) ===")
    print(f"{'tag':12} {'server':>7} {'judge_mean':>11} {'n':>4}")
    rows_sorted = sorted(meta.items(), key=lambda kv: -kv[1]["server"])
    for tag, m in rows_sorted:
        scores = [best[(tag, rid)] for rid in rids if (tag, rid) in best]
        jm = sum(scores) / len(scores) if scores else float("nan")
        summary["anchors"][tag] = {"server": m["server"], "judge_mean": jm, "n": len(scores)}
        print(f"{tag:12} {m['server']:>7} {jm:>11.2f} {len(scores):>4}")

    # Проверка монотонности ранжирования
    server_order = [t for t, _ in rows_sorted]
    judge_means = [summary["anchors"][t]["judge_mean"] for t in server_order]
    inversions = sum(1 for i in range(len(judge_means))
                     for j in range(i + 1, len(judge_means))
                     if judge_means[i] < judge_means[j])
    print(f"\nинверсий ранжирования: {inversions} (0 = идеальный порядок)")
    summary["rank_inversions"] = inversions
    summary["server_order"] = server_order

    # Per-subject на топ-якоре (должен ловить knowledge-gap: лит-ра/рус низко)
    top_tag = server_order[0]
    bysubj = defaultdict(list)
    for rid in rids:
        if (top_tag, rid) in best and rid in subj_of:
            bysubj[subj_of[rid]].append(best[(top_tag, rid)])
    print(f"\n=== per-subject на топ-якоре '{top_tag}' (knowledge-gap проверка) ===")
    for subj in sorted(bysubj, key=lambda s: sum(bysubj[s]) / len(bysubj[s])):
        vals = bysubj[subj]
        m = sum(vals) / len(vals)
        summary["per_subject"][subj] = {"mean": m, "n": len(vals)}
        print(f"  {subj:12} {m:6.2f}  (n={len(vals)})")

    sum_path = out_dir / f"summary_{args.run_tag}.json"
    json.dump(summary, open(sum_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nwrote {sum_path}")
    print(f"raw   {raw_path}")


if __name__ == "__main__":
    main()
