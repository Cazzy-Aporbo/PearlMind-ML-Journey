
#!/usr/bin/env python3
import argparse, json, re
import pandas as pd
from pathlib import Path
from charts import bar_counts, bar_means, hist_series
from theme import PALETTE

# Default: look for dolly file in THIS folder
DEFAULT_DATA = Path(__file__).parent / "databricks-dolly-15k.jsonl"
ASSETS_DIR = Path(__file__).parent / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

def load_dolly(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    df = pd.DataFrame(rows)
    expected = {"instruction","context","response","category"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df

def basic_lengths(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["instruction_len"] = df["instruction"].astype(str).str.len()
    df["response_len"] = df["response"].astype(str).str.len()
    return df

def per_category_counts(df: pd.DataFrame):
    return df["category"].value_counts().to_dict()

def per_category_means(df: pd.DataFrame):
    g = df.groupby("category").agg(inst_len_mean=("instruction_len","mean"),
                                   resp_len_mean=("response_len","mean"))
    return g

def top_tokens(texts, k=20):
    freq = {}
    for t in texts:
        for tok in re.findall(r"[A-Za-z']+", str(t).lower()):
            freq[tok] = freq.get(tok, 0) + 1
    return sorted(freq.items(), key=lambda x: x[1], reverse=True)[:k]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=str(DEFAULT_DATA),
                    help="Path to databricks-dolly-15k.jsonl (defaults to file in this folder).")
    args = ap.parse_args()
    data_path = Path(args.data)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            f"Tip: put databricks-dolly-15k.jsonl next to analyze.py or pass --data /path/to/file"
        )

    df = load_dolly(data_path)
    df = basic_lengths(df)

    summary = {
        "num_records": int(len(df)),
        "columns": list(df.columns),
        "null_counts": df.isnull().sum().to_dict(),
        "instruction_len_mean": float(df["instruction_len"].mean()),
        "response_len_mean": float(df["response_len"].mean()),
        "instruction_len_median": float(df["instruction_len"].median()),
        "response_len_median": float(df["response_len"].median()),
    }
    (ASSETS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # Charts
    hist_series(df["instruction_len"], "Instruction Length Distribution", ASSETS_DIR / "instruction_length_dist.png",
                color=PALETTE["pink"], edge=PALETTE["purple_text"])
    hist_series(df["response_len"], "Response Length Distribution", ASSETS_DIR / "response_length_dist.png",
                color=PALETTE["lavender"], edge=PALETTE["teal_edge"])

    counts = per_category_counts(df)
    bar_counts(counts, "Records per Category", ASSETS_DIR / "category_counts.png",
               color_key="lavender", edge_key="lavender_edge")

    means = per_category_means(df)
    bar_means(means["inst_len_means".replace("means","_mean")].to_dict(), "Mean Instruction Length by Category",
              ASSETS_DIR / "inst_len_means.png", color_key="teal", edge_key="teal_edge", ylabel="Chars")
    bar_means(means["resp_len_means".replace("means","_mean")].to_dict(), "Mean Response Length by Category",
              ASSETS_DIR / "resp_len_means.png", color_key="blue", edge_key="teal_edge", ylabel="Chars")

    # Optional token peeks
    sample_creative = df[df["category"].str.contains("creative", case=False, na=False)].head(200)
    sample_closed   = df[df["category"].str.contains("closed",   case=False, na=False)].head(200)
    (ASSETS_DIR / "creative_top_tokens.json").write_text(json.dumps(top_tokens(sample_creative["response"], 25), indent=2))
    (ASSETS_DIR / "closedqa_top_tokens.json").write_text(json.dumps(top_tokens(sample_closed["response"], 25), indent=2))

    print("✅ Analysis complete. See assets/: instruction_length_dist.png, response_length_dist.png, category_counts.png, inst_len_means.png, resp_len_means.png")

if __name__ == "__main__":
    main()
