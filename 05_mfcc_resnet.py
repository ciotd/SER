import argparse
import os
from pathlib import Path
import json
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

class EmotionRESNET(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.model = models.resnet18(weights=None)  # no pretrained
        # 1-ch spectrogram
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class EmotionDataset(Dataset):
    def __init__(self, file_paths, labels,
                 sr=16000, duration=3, n_fft=512, hop_length=160, win_length=400, n_mels=64, n_mfcc=40):
        self.file_paths = file_paths
        self.labels = labels
        self.sr = sr
        self.duration = duration
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.max_len = sr * duration
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc

    def __len__(self):
        return len(self.file_paths)

    @staticmethod
    def _minmax_to_pm1(A: np.ndarray) -> np.ndarray:
        mn, mx = float(A.min()), float(A.max())
        if mx - mn < 1e-8:
            return np.zeros_like(A, dtype=np.float32)
        A = (A - mn) / (mx - mn + 1e-8)
        return (2.0 * A - 1.0).astype(np.float32)

    def __getitem__(self, index):
        fp = self.file_paths[index]
        y, _ = librosa.load(fp, sr=self.sr)

        # pad/clip 3s
        if len(y) < self.max_len:
            y = np.pad(y, (0, self.max_len - len(y)))
        else:
            y = y[:self.max_len]

        # MFCC (Mel -> dB -> DCT)
        M = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_fft=self.n_fft,
            hop_length=self.hop_length, win_length=self.win_length,
            n_mels=self.n_mels, center=True, power=2.0
        )
        M_db = librosa.power_to_db(M, ref=np.max)
        
        # MFCC-dct
        MFCC = librosa.feature.mfcc(
            S=M_db, sr=self.sr, n_mfcc=self.n_mfcc, dct_type=2, lifter=0
        )

        A = self._minmax_to_pm1(MFCC)        # [-1, 1] 정규화
        x = torch.tensor(A, dtype=torch.float32).unsqueeze(0)  # (1, n_mfcc, T)
        y_lbl = torch.tensor(self.labels[index], dtype=torch.long)
        return x, y_lbl


# CSV DataLoaders
def prepare_dataloaders_from_csv(
    split_dir, batch_size=32, sr=16000, duration=3, n_fft=512, hop_length=160, win_length=400, n_mels=64, n_mfcc=40
):
    split_dir = Path(split_dir)
    tr = pd.read_csv(split_dir / "train.csv")
    va = pd.read_csv(split_dir / "val.csv")
    te = pd.read_csv(split_dir / "test.csv")

    # label_map / id2label
    id2label = None
    meta_path = split_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            lm = meta.get("label_map")
            if lm:
                id2label = {int(v): k for k, v in lm.items()}
        except Exception:
            pass
    if id2label is None and "label" in tr.columns and "label_id" in tr.columns:
        pairs = pd.concat([tr, va, te])[["label", "label_id"]].drop_duplicates()
        id2label = {int(r["label_id"]): r["label"] for _, r in pairs.iterrows()}

    def df_to_ds(df):
        paths = df["filepath"].astype(str).tolist()
        labels = df["label_id"].astype(int).tolist()
        return EmotionDataset(paths, labels, sr=sr, duration=duration,
                              n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_mels=n_mels, n_mfcc=n_mfcc)

    train_ds = df_to_ds(tr)
    val_ds   = df_to_ds(va)
    test_ds  = df_to_ds(te)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    num_classes = len(set(pd.concat([tr["label_id"], va["label_id"], te["label_id"]]).astype(int)))
    # class_names: 0..N-1 기준 정렬. id2label 없으면 문자열로 대체.
    if id2label is None:
        class_names = [str(i) for i in range(num_classes)]
    else:
        # 보장: 0..N-1 모두 존재한다고 가정. 누락 시 문자열로 대체.
        class_names = [id2label.get(i, str(i)) for i in range(num_classes)]

    return train_loader, val_loader, test_loader, num_classes, class_names

# Train / Eval
def Train_model(model, train_loader, val_loader, num_epochs=5, lr=1e-3,
                device="cuda", save_path="./model/best_model_mfcc_resnet.pth"):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for X, y in tqdm(train_loader, desc=f"train {epoch+1}/{num_epochs}"):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            _, pred = torch.max(outputs, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            total_loss += loss.item() * y.size(0)

        train_acc = correct / total

        # val
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for X, y in tqdm(val_loader, desc="val"):
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                _, pred = torch.max(outputs, 1)
                v_correct += (pred == y).sum().item()
                v_total += y.size(0)
        val_acc = v_correct / v_total
        avg_loss = total_loss / total

        print(f"Epoch {epoch+1}/{num_epochs} | Loss {avg_loss:.4f} | Train {train_acc:.4f} | Val {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f">>> best model updated -> val_acc = {val_acc:.4f}")

    return model

def evaluate_model(model, test_loader, device="cuda", class_names=None):
    model.to(device)
    model.eval()
    all_preds, all_labels = [], []

    correct, total = 0, 0
    with torch.no_grad():
        for X, y in tqdm(test_loader, desc="test"):
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, pred = torch.max(outputs, 1)
            all_preds.extend(pred.detach().cpu().numpy())
            all_labels.extend(y.detach().cpu().numpy())
            correct += (pred == y).sum().item()
            total   += y.size(0)
            
    all_preds = np.asarray(all_preds)
    all_labels = np.asarray(all_labels)
    accuracy = correct / total

    print(f"Accuracy : {accuracy}")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


# Main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, choices=["train", "test"], required=True)
    ap.add_argument("--split_dir", type=str, default="./dataset", help="train/val/test CSV 위치")
    ap.add_argument("--checkpoint", type=str, default="./model/best_model_mfcc_resnet.pth")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--gpu", type=int, default=0)


    # MFCC params
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--duration", type=int, default=3)
    ap.add_argument("--n_fft", type=int, default=512)
    ap.add_argument("--hop_length", type=int, default=160)
    ap.add_argument("--win_length", type=int, default=400)
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--n_mfcc", type=int, default=40)


    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, test_loader, num_classes, class_names = prepare_dataloaders_from_csv(
        split_dir=args.split_dir,
        batch_size=args.batch_size,
        sr=args.sr, duration=args.duration,
        n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.win_length
    )

    model = EmotionRESNET(num_classes=num_classes)

    if args.mode == "train":
        Train_model(model, train_loader, val_loader, num_epochs=args.epochs,
                    device=device, save_path=args.checkpoint)

    else:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        evaluate_model(model, test_loader, device=device, class_names=class_names)
