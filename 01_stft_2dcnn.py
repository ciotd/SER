import argparse
import os
from tqdm import tqdm
import glob
import librosa
import numpy as np
import json
from pathlib import Path
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 추가: sklearn 지표
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)

# ======================= 2D CNN 모델 정의 ================================
class Emotion_2DCNN(nn.Module):
    def __init__(self, num_classes, input_shape):
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

        # dummy 입력으로 flatten 크기 자동 산출
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

# ====================== Dataset (전처리 포함)========================
class EmotionDataset(Dataset):
    def __init__(self, file_paths, labels, sr, duration, n_fft, hop_length, win_length):
        self.file_paths = file_paths
        self.labels = labels
        self.sr = sr
        self.duration = duration
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.max_len = sr * duration

    def __len__(self):
        return len(self.file_paths)

    # ------ STFT 전처리 로직 -------
    def __getitem__(self, index):
        file_path = self.file_paths[index]
        label = self.labels[index]

        # 1) load
        y, _ = librosa.load(file_path, sr=self.sr)

        # 2) pad/clip to fixed length
        if len(y) < self.max_len:
            y = np.pad(y, (0, self.max_len - len(y)))
        else:
            y = y[:self.max_len]

        # 3) STFT -> |.| -> dB
        spec = np.abs(librosa.stft(
            y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=True
        ))
        spec_db = librosa.amplitude_to_db(spec, ref=np.max)

        # 4) normalize to [-1,1]
        mn, mx = spec_db.min(), spec_db.max()
        spec_db = (spec_db - mn) / (mx - mn + 1e-8)
        spec_db = 2 * spec_db - 1

        # 5) tensor
        spec_tensor = torch.tensor(spec_db, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return spec_tensor, label_tensor

# ------ csv 기반 데이터 로더 -------
def prepare_dataloaders_from_csv(split_dir, batch_size, sr, duration, n_fft, hop_length, win_length):
    split_dir = Path(split_dir)  # train.csv, val.csv, test.csv 파일이 있는 디렉토리 경로
    tr = pd.read_csv(split_dir / "train.csv")
    va = pd.read_csv(split_dir / "val.csv")
    te = pd.read_csv(split_dir / "test.csv")

    meta_path = split_dir / "meta.json"
    label_map = None  # label_map 초기화
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            label_map = meta.get("label_map", None)
        except Exception:
            label_map = None

    if label_map is None:
        print("[오류] label_map 이 없습니다... (지표 계산/출력에는 필수는 아님)")

    id2label = {v: k for k, v in label_map.items()} if label_map else None

    def df_to_ds(df):
        paths = df["filepath"].astype(str).tolist()  # 파일 경로 리스트
        labels = df["label_id"].astype(int).tolist()  # 라벨 정답값 리스트
        return EmotionDataset(
            paths,
            labels,
            sr=sr,
            duration=duration,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )

    train_ds = df_to_ds(tr)
    val_ds = df_to_ds(va)
    test_ds = df_to_ds(te)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    num_classes = len(set(pd.concat([tr["label_id"], va["label_id"], te["label_id"]]).astype(int).tolist()))
    if id2label is None:
        class_names = [str(i) for i in range(num_classes)]
    else:
        # 보장: 0..N-1 모두 존재한다고 가정. 누락 시 문자열로 대체.
        class_names = [id2label.get(i, str(i)) for i in range(num_classes)]    
    return train_loader, val_loader, test_loader, num_classes, class_names

# ================= 시간/주파수 프레임 추정 (STFT 기준) ==============
def estimate_time_freq(sr, duration, n_fft, hop_length, win_length):
    y = np.zeros(sr * duration, dtype=np.float32)
    spec = np.abs(librosa.stft(
        y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=True
    ))
    F, T = spec.shape  # F=freq bins, T=time frames
    print(f"spec.shape: (freq={F}, time={T})")
    return F, T


# ========================= Train Mode =====================
def train_model(model, train_loader, val_loader, num_epochs, device, save_path, lr=1e-3,
                out_dir=None, num_classes=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for X, y in tqdm(train_loader, desc=f"training (epoch {epoch+1})..."):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            # 예측 라벨
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
            total_loss += loss.item() * y.size(0)

        train_acc = correct / total

        # ---------- validation ----------
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
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # ---- checkpoint(체크포인트) 저장 및 검증 지표 CSV -----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(">>> best model updated.")

    return model

# ====================== Test Mode ==========================
def evaluate_model(model, test_loader, device, class_names):
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



# ==================================== 메인 루프 ===================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True, help="학습 모드일 땐 train, 평가 모드일 땐 test")
    parser.add_argument("--split_dir", type=str, default="./dataset_splits")
    parser.add_argument("--checkpoint", type=str, default="./model/best_model_stft_2dcnn_0902.pth", help="저장된 모델 불러오기 (경로 적기)")
    parser.add_argument("--epochs", type=int, default=5, help="학습 epoch 수")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--out_dir", type=str, default="./outputs", help="CSV 저장 폴더")

    # stft parameter (모델 input_shape 계산과 dataset 동시에 사용)
    parser.add_argument("--sr", type=int, default=16000)  # 16kHz
    parser.add_argument("--duration", type=int, default=3)  # 3초
    parser.add_argument("--n_fft", type=int, default=512)
    parser.add_argument("--hop_length", type=int, default=160)
    parser.add_argument("--win_length", type=int, default=400)
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # DataLoader
    train_loader, val_loader, test_loader, num_classes, class_names = prepare_dataloaders_from_csv(
        split_dir=args.split_dir,
        batch_size=args.batch_size,
        sr=args.sr,
        duration=args.duration,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length
    )

    # calculate input_shape from STFT (freq bins, time frames)
    F_est, T_est = estimate_time_freq(
        sr=args.sr, duration=args.duration,
        n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.win_length
    )

    # model 정의
    model = Emotion_2DCNN(num_classes=num_classes, input_shape=(1, F_est, T_est))

    print("======================================> step 1 완료")

    if args.mode == "train":
        train_model(
            model, train_loader, val_loader,
            num_epochs=args.epochs,                    # FIX: args.epochs 사용
            device=device,
            save_path=args.checkpoint,
            out_dir=args.out_dir,
            num_classes=num_classes
        )
    else:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        evaluate_model(model, test_loader, device=device, class_names=class_names)
