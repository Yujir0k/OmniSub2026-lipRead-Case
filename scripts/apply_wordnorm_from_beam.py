import argparse
from pathlib import Path

import pandas as pd


def apply_wordnorm(text: str) -> str:
    # Exact rule used to produce submission_run_char_warmstart_beam30_wordnorm.csv
    token_map = {"e": "the", "o": "of"}
    toks = [token_map.get(t, t) for t in str(text).split()]

    # Limit consecutive repeats (e.g. "a a a a" -> "a a")
    out = []
    prev = None
    cnt = 0
    for t in toks:
        if t == prev:
            cnt += 1
        else:
            prev = t
            cnt = 1
        if cnt <= 2:
            out.append(t)
    return " ".join(out).strip()


def main(args):
    inp = Path(args.input_csv).resolve()
    out = Path(args.output_csv).resolve()

    df = pd.read_csv(inp)
    if "path" not in df.columns or "transcription" not in df.columns:
        raise ValueError("CSV must contain columns: path, transcription")

    df["transcription"] = df["transcription"].astype(str).map(apply_wordnorm)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"saved: {out}")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser("Apply word normalization to beam submission")
    p.add_argument("--input-csv", type=str, required=True)
    p.add_argument("--output-csv", type=str, required=True)
    main(p.parse_args())
