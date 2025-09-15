import argparse
import os
from tqdm import tqdm
import glob
import librosa
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models

# scikitlearn 라이브러리에서 데이터 분할 해주는 함수
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader

# 2D CNN 딥러닝 모델 네트워크 -> 간단하게 직접 만들어 본 것.
class EmotionRESNET(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        #ResNet18 불러오기(사전학습 가중치는 X)
        self.model = models.resnet18(weights=None)

        # 1채널 입력 스펙트로그램을 받을 수 있게 변경한 것. 원래는 3채널이 들어오게 되어있대
        self.model.conv1 = nn.Conv2d(1,64, kernel_size=7, stride=2, padding=3, bias=False)

        # 출력 차원을 우리가 풀려는 5개의 크래스에 맞게 바꾼 것.
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self,x):
        return self.model(x)

# EmotionDataset -> 데이터셋 정의
class EmotionDataset(Dataset):
    def __init__(self, file_paths, labels, sr=16000, duration=3, n_fft=512, hop_length=160, win_length=400):
        self.file_paths = file_paths
        self.labels = labels
        self.sr = sr
        self.duration = duration
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.max_len = sr * duration # 고정 길이를 준다 3초 동안...
    
    def __len__(self):
        return len(self.file_paths) # 데이터셋 샘플 갯수가 반환됨.
    
    def __getitem__(self, index):
        file_path = self.file_paths[index]
        label = self.labels[index]

        # 1) 오디오 로드
        y, sr = librosa.load(file_path, sr=self.sr)

        # 2) 패딩/클리핑 (고정 길이 3초)
        if len(y) < self.max_len:
            y = np.pad(y, (0, self.max_len - len(y))) # 길이가 작으면 그 뒤로 다 0으로 채운다 이 말이잖아.
        else:
            y = y[:self.max_len] # 길이가 3초 분량보다 더 길면 그냥 3초 분량까지만 자른다는 거고.

        # 3) STFT -> magnitude -> dB
        stft = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
        spec = np.abs(stft)
        spec_db = librosa.amplitude_to_db(spec, ref=np.max)

        # 4) Normalize [-1,1]
        spec_db = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min())
        spec_db = 2*spec_db -1

        # 5) Tensor 변환 (채널 1개 추가해야 함) 
        # # -> 여기서 spec 이미지랑 라벨 따로 만드는 건 나도 아직 이해 못함.
        spec_tensor = torch.tensor(spec_db, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return spec_tensor, label_tensor

# 데이터 준비 함수
def prepare_dataloaders(root_dir, test_size=0.2, val_size=0.1, batch_size=32):
    emotions = sorted(os.listdir(root_dir)) # 기쁨, 슬픔, 분노, etc 감정 카테고리 별로 디렉토리가 정리되어 있음
    file_paths, labels = [], []
    label_map = {emo: idx for idx, emo in enumerate(emotions)}

    # 디렉토리 명으로 데이터-라벨 리스트 만들기
    for emo in tqdm(emotions):
        files = glob.glob(os.path.join(root_dir, emo, "*.wav"))
        for f in files:
            file_paths.append(f)
            labels.append(label_map[emo])
        
    # train/val/test split (고정 seed=42)
    train_files, test_files, train_labels, test_labels = train_test_split(file_paths, labels, test_size=test_size, random_state=42, stratify=labels)
    train_files, val_files, train_labels, val_labels = train_test_split(train_files, train_labels, test_size=val_size, random_state=42, stratify=train_labels)

    train_dataset = EmotionDataset(train_files, train_labels)
    val_dataset = EmotionDataset(val_files, val_labels)
    test_dataset = EmotionDataset(test_files, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# 학습 함수
def Train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-3, device="cuda", save_path="./model/best_model_stft_resnet.pth"):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Train 모드 진입
        model.train()
        total_loss, correct, total = 0, 0, 0
        for X, y in train_loader: # dataloader가 batch 단위로 학습 데이터를 던져주는거임~
            X, y = X.to(device), y.to(device) # 그리고 gpu로 데이터들을 이동시켜
            optimizer.zero_grad() # 이전 batch에서 쌓인 gradient 처음에 초기화 시켜주고
            outputs = model(X) # CNN forward 연산을 해 -> 그럼 예측 결과가 나와
            loss = criterion(outputs, y) # 예측 결과와 실제 정답을 비교해서 손실값 계산해
            loss.backward() # 손실에 대한 gradient를 계산해
            optimizer.step() # 계산된 gradient로 파라미터를 업데이트 해

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1) # 예측된 클래스를 가장 큰 값의 index로 넣는다..
            correct += (predicted == y).sum().item()
            total += y.size(0)

        train_acc = correct / total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == y).sum().item()
                val_total += y.size(0)
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # best.pt 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f">>> best model updated -> val_acc = {val_acc:.4f}")

    return model

# 평가 함수
def evaluate_model(model, test_loader, device="cuda", class_names=None):
    model.to(device)
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Accuracy, Precision, Recall, F1
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    # Confusion Matrix
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
            


# 메인 루프
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True, help="학습 모드일 땐 train, 평가 모드일 땐 test")
    parser.add_argument("--dataset", type=str, help="dataset path", default=r"E:\03_datasets\datasets_aihub_actor")
    parser.add_argument("--checkpoint", type=str, default="./model/best_model_stft_resnet.pth", help="저장된 모델 불러오기 (경로 적기)")
    parser.add_argument("--epochs", type=int, default=10, help="학습 epoch 수")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    root_dir = args.dataset
    class_names = sorted(os.listdir(args.dataset))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, test_loader = prepare_dataloaders(args.dataset, batch_size=args.batch_size)
    model = EmotionRESNET(num_classes=len(class_names))

    if args.mode == "train":
        model = Train_model(model, train_loader, val_loader, num_epochs=args.epochs, device=device, save_path=args.checkpoint)

    elif args.mode == "test":
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        evaluate_model(model, test_loader, device=device, class_names=class_names)
