import argparse
import os
from tqdm import tqdm
import glob
import librosa
import numpy as np
import json, hashlib
from pathlib import Path
import pandas as pd

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report


# 모델
class Emotion_2DCNN(nn.Module):
    def __init__(self, num_classes=5, input_shape=(1, 257, 301)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        with torch.no_grad():
            dummy = torch.zeros((1, *input_shape))
            out = self.features(dummy)
            flatten_size = out.view(1, -1).size(1)

        self.fc1 = nn.Linear(flatten_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


# Dataset
class EmotionDataset(Dataset):
    def __init__(self, file_paths, labels,
                 sr=16000, duration=3, n_fft=512, hop_length=160, win_length=400, n_mels=128):
        self.file_paths = file_paths
        self.labels = labels
        self.sr = sr
        self.duration = duration
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.max_len = sr * duration

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        label = self.labels[index]

        y, _ = librosa.load(file_path, sr=self.sr)
        if len(y) < self.max_len:
            y = np.pad(y, (0, self.max_len - len(y)))
        else:
            y = y[:self.max_len]

        mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, n_mels=self.n_mels, power=2.0)
        spec_db = librosa.power_to_db(mel,ref=np.max)
        spec_db = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min() + 1e-8)
        spec_db = 2 * spec_db - 1

        spec_tensor = torch.tensor(spec_db, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return spec_tensor, label_tensor


# Split 고정 함수
def _scan_dataset(root_dir):
    root = Path(root_dir)
    emotions = sorted([d.name for d in root.iterdir() if d.is_dir()])
    label_map = {emo: i for i, emo in enumerate(emotions)}
    file_paths, labels = [], []
    for emo in emotions:
        for f in sorted(glob.glob(str(root / emo / "*.wav"))):
            file_paths.append(f)
            labels.append(label_map[emo])
    return file_paths, labels, label_map, emotions

def _hash_split_inputs(file_paths, labels, test_size, val_size, seed, emotions_sorted):
    h = hashlib.sha256()
    for p, l in zip(file_paths, labels):
        h.update(p.encode("utf-8"))
        h.update(str(l).encode("utf-8"))
    h.update(f"{test_size}_{val_size}_{seed}_{';'.join(emotions_sorted)}".encode("utf-8"))
    return h.hexdigest()

def make_or_load_split(root_dir, split_path, test_size=0.2, val_size=0.1, seed=42, resplit=False):
    file_paths, labels, label_map, emotions = _scan_dataset(root_dir)
    split_path = Path(split_path)

    if split_path.exists() and not resplit:
        try:
            saved = json.loads(split_path.read_text(encoding="utf-8"))
            if saved["meta"]["hash"] == _hash_split_inputs(file_paths, labels, test_size, val_size, seed, emotions):
                return (saved["train"]["files"], saved["train"]["labels"]), \
                       (saved["val"]["files"],   saved["val"]["labels"]), \
                       (saved["test"]["files"],  saved["test"]["labels"]), label_map
        except Exception:
            pass

    # 새로 분할
    tr_f, te_f, tr_l, te_l = train_test_split(
        file_paths, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    tr_f, va_f, tr_l, va_l = train_test_split(
        tr_f, tr_l, test_size=val_size, random_state=seed, stratify=tr_l
    )

    meta_hash = _hash_split_inputs(file_paths, labels, test_size, val_size, seed, emotions)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps({
        "meta": {"root_dir": str(root_dir), "test_size": test_size,
                 "val_size": val_size, "seed": seed, "emotions": emotions, "hash": meta_hash},
        "label_map": label_map,
        "train": {"files": tr_f, "labels": tr_l},
        "val":   {"files": va_f, "labels": va_l},
        "test":  {"files": te_f, "labels": te_l},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    return (tr_f, tr_l), (va_f, va_l), (te_f, te_l), label_map

def prepare_dataloaders_from_csv(split_dir, batch_size=32, n_mels=128):
    split_dir = Path(split_dir)
    tr = pd.read_csv(split_dir / "train.csv")
    va = pd.read_csv(split_dir / "val.csv")
    te = pd.read_csv(split_dir / "test.csv")

    # label_map/id2label 구성
    meta_path = split_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        label_map = meta.get("label_map", None)  # {"anger":0, ...}
        if label_map is None:
            # CSV에서 유추
            pairs = pd.concat([tr, va, te])[["label", "label_id"]].drop_duplicates()
            label_map = {row["label"]: int(row["label_id"]) for _, row in pairs.iterrows()}
    else:
        # CSV에서 유추
        pairs = pd.concat([tr, va, te])[["label", "label_id"]].drop_duplicates()
        label_map = {row["label"]: int(row["label_id"]) for _, row in pairs.iterrows()}

    id2label = {v: k for k, v in label_map.items()}

    # Dataset
    def df_to_ds(df):
        filepaths = df["filepath"].tolist()
        labels = df["label_id"].astype(int).tolist()
        return EmotionDataset(filepaths, labels, n_mels=n_mels)

    train_ds = df_to_ds(tr)
    val_ds   = df_to_ds(va)
    test_ds  = df_to_ds(te)

    # DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # 클래스 수
    num_classes = len(set(pd.concat([tr["label_id"], va["label_id"], te["label_id"]]).astype(int).tolist()))
    return train_loader, val_loader, test_loader, label_map, id2label, num_classes


# Train & Evaluate
def Train_model(model, train_loader, val_loader, num_epochs=5, lr=1e-3,
                device="cuda", save_path="./model/best_model_mel_2dcnn.pth"):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        print(f"epoch: {epoch}") #debug
        for X, y in tqdm(train_loader, desc="training..."):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            bs = y.size(0)
            total_loss += loss.item() * bs
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += bs

        train_acc = correct / total

        # validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for X, y in tqdm(val_loader, desc="validation..."):
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == y).sum().item()
                val_total += y.size(0)
        val_acc = val_correct / val_total

        avg_loss = total_loss / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f">>> best model updated -> val_acc = {val_acc:.4f}")

    return model

def evaluate_model(model, test_loader, id2label, csv_path, device="cuda"):
    """
    - 정확도, precision/recall/f1(macro/weighted) 출력
    - 예측 결과 CSV 저장: [filepath, true_id, pred_id, true_label, pred_label]
    """
    model.to(device)
    model.eval()

    all_preds, all_trues, all_paths = [], [], []
    offset = 0  # 파일 경로 매칭용
    file_paths = test_loader.dataset.file_paths  # shuffle=False 라서 순서 보장

    with torch.no_grad():
        for X, y in tqdm(test_loader, desc="evaluating..."):
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            preds = torch.argmax(outputs, dim=1)

            bs = y.size(0)
            batch_paths = file_paths[offset:offset+bs]
            offset += bs

            all_preds.extend(preds.detach().cpu().tolist())
            all_trues.extend(y.detach().cpu().tolist())
            all_paths.extend(batch_paths)

    # 지표 계산
    acc = accuracy_score(all_trues, all_preds)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(all_trues, all_preds, average="macro", zero_division=0)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(all_trues, all_preds, average="weighted", zero_division=0)

    print("\n=== Test Metrics ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro     - Precision: {prec_m:.4f}, Recall: {rec_m:.4f}, F1: {f1_m:.4f}")
    print(f"Weighted  - Precision: {prec_w:.4f}, Recall: {rec_w:.4f}, F1: {f1_w:.4f}")

    # 클래스별 리포트(선택 출력)
    try:
        target_names = [id2label[i] for i in sorted(set(all_trues + all_preds))]
        print("\nPer-class report:")
        print(classification_report(all_trues, all_preds, target_names=target_names, zero_division=0))
    except Exception:
        # id2label이 불완전한 경우에도 안전하게 넘어감
        pass

    # CSV 저장
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "true_id", "pred_id", "true_label", "pred_label"])
        for p, t, pr in zip(all_paths, all_trues, all_preds):
            writer.writerow([p, t, pr, id2label.get(t, str(t)), id2label.get(pr, str(pr))])

    print(f"\nSaved prediction CSV → {csv_path}")

def estimate_time_frames(sr=16000, duration=3, n_fft=512, hop_length=160, win_length=400, n_mels=128):
    y = np.zeros(sr*duration, dtype=np.float32)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        n_mels=n_mels, power=2.0
    )
    return mel.shape[1]  # time frames


# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True)
    parser.add_argument("--dataset", type=str, default="storage1/SER/datasets_aihub_actor")
    parser.add_argument("--checkpoint", type=str, default="./model/best_model_mel_2dcnn.pth")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)

    # CSV split 디렉터리
    parser.add_argument("--split_dir", type=str, default="./dataset_splits",
                        help="train.csv/val.csv/test.csv가 있는 디렉토리")

    # 단일 GPU 선택
    parser.add_argument("--gpu", type=int, default=0, help="사용할 GPU ID (0부터 시작)")

    # 평가 결과 CSV 경로
    parser.add_argument("--metrics_csv", type=str, default="./results/mel_2dcnn_results.csv",
                        help="테스트셋 예측 결과를 저장할 CSV 경로")

    args = parser.parse_args()

    # 디바이스 선택
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # DataLoader 준비(분할은 CSV에서 고정 로드)
    train_loader, val_loader, test_loader, label_map, id2label, num_classes = prepare_dataloaders_from_csv(
        split_dir=args.split_dir,
        batch_size=args.batch_size,
        n_mels=128
    )

    # 입력 크기 동기화(시간 프레임 동적 계산)
    T_est = estimate_time_frames(sr=16000, duration=3, n_fft=512, hop_length=160, win_length=400, n_mels=128)

    # 모델 생성
    model = Emotion_2DCNN(num_classes=num_classes, input_shape=(1, 128, T_est))

    if args.mode == "train":
        Train_model(model, train_loader, val_loader,
                    num_epochs=args.epochs, device=device, save_path=args.checkpoint)
    else:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        evaluate_model(model, test_loader, id2label, args.metrics_csv, device=device)