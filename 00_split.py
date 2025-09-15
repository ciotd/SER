import os, glob, json, hashlib
from pathlib import Path
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def scan_dataset(dataset):
    root_path = Path(dataset)
    emotions = sorted([ d.name for d in root_path.iterdir() if d.is_dir()]) # 감정 클래스
    label_map = {emo: i for i, emo in enumerate(emotions)}
    rows = []
    for emo in emotions:
        for f in sorted(glob.glob(str(root_path/emo/"*.wav"))):
            rows.append({"filepath": os.path.abspath(f), "label":emo, "label_id": label_map[emo]})
    return pd.DataFrame(rows), label_map, emotions
    
def hash_inputs(df, seed): # 이거 해시테이블은 진짜 뭔지 모르겠음ㅠㅠㅠ 왜 필요한거야..
    h = hashlib.sha256()
    for _, r in df.iterrows():
        h.update(r["filepath"].encode()); h.update(str(r["label_id"]).encode())
    h.update(f"seed={seed}".encode())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out_dir", default="./dataset_splits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--format", choices=["csv","json"], default="csv")
    args = ap.parse_args()

    df, label_map, emotions = scan_dataset(args.dataset) # 첫번째 함수 호출
    assert len(df) > 0, "(오류) wav files 없음"

    # 8:1:1 split with stratify
    train_df, tmp_df = train_test_split(df, test_size=0.2, random_state=args.seed, stratify=df["label_id"])
    val_df, test_df = train_test_split(tmp_df, test_size=0.5, random_state=args.seed, stratify=tmp_df["label_id"])

    os.makedirs(args.out_dir, exist_ok=True)
    meta = {
        "dataset_root" : os.path.abspath(args.dataset),
        "label_map" : label_map,
        "hash" : hash_inputs(df, args.seed), # hash_inputs 함수 
        "counts" : {"train": len(train_df), "val":len(val_df), "test":len(test_df)},

    }
    
    if args.format == "csv":
        train_df.to_csv(Path(args.out_dir)/"train.csv", index=False)
        val_df.to_csv(Path(args.out_dir)/"val.csv", index=False)
        test_df.to_csv(Path(args.out_dir)/"test.csv", index=False)
        with open(Path(args.out_dir)/"meta.json","w",encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print("train.csv, val.csv, test.csv, meta.json 파일이 전부 저장되었습니다.")
    else:
        out = {
            "meta": meta,
            "train": train_df.to_dict(orient="records"),
            "val":   val_df.to_dict(orient="records"),
            "test":  test_df.to_dict(orient="records"),
        }
        with open(Path(args.out_dir)/"splits.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("Saved: splits.json")
        
if __name__ == "__main__":
    main()