import os
import random
import shutil

SOURCE_PATH = r"E:\03_datasets\datasets_aihub_actor"
TARGET_PATH = r"E:\03_datasets\00_samples"
categories = ["기쁨", "슬픔", "분노", "불안", "중립"]

if __name__ == "__main__":
    # 1. 대상 폴더 생성
    for cat in categories:
        os.makedirs(os.path.join(TARGET_PATH, cat), exist_ok=True)

    # 2. 전체 wav 파일 수집
    all_files = []
    for root, dirs, files in os.walk(SOURCE_PATH):
        for file in files:
            if file.lower().endswith(".wav"):
                all_files.append(os.path.join(root, file))

    print(f"총 파일 개수: {len(all_files)}")

    # 3. 무작위 1% 샘플링
    sample_size = max(1, int(len(all_files) * 0.001))  # 최소 1개 보장
    sampled_files = random.sample(all_files, sample_size)

    # 4. 파일 복사
    for file_path in sampled_files:
        # 카테고리 추출
        parts = file_path.split(os.sep)
        cat = next((c for c in categories if c in parts), None)

        if cat:
            dest_path = os.path.join(TARGET_PATH, cat, os.path.basename(file_path))
            shutil.copy2(file_path, dest_path)

    print(f"복사 완료: {len(sampled_files)}개")

