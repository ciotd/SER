import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

SOURCE_PATH = r"E:\03_산학연_음성감정인식\015.감성 및 발화 스타일별 음성합성 데이터\01.데이터\1.Training\원천데이터\TS2\TS2"
TARGET_PATH = r"E:\03_datasets\datasets_aihub_actor"
categories = ["기쁨", "슬픔", "분노", "불안", "중립"]


def copy_file(file_path, new_name):
    try:
        root = os.path.dirname(file_path)
        parts = root.split(os.sep)
        emotion_folder = None
        for p in parts:
            if any(cat in p for cat in categories):
                emotion_folder = p
                break

        if emotion_folder:
            for cat in categories:
                if cat in emotion_folder:
                    dest_dir = os.path.join(TARGET_PATH, cat)
                    dest_path = os.path.join(dest_dir, new_name)
                    shutil.copy2(file_path, dest_path)
                    return True
        return False
    except Exception as e:
        return f"Error: {file_path}, {e}"


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

    existing_files = []
    for cat in categories:
        cat_path = os.path.join(TARGET_PATH, cat)
        if os.path.exists(cat_path):
            for f in os.listdir(cat_path):
                if f.lower().endswith(".wav"):
                    existing_files.append(f)

    start_idx = len(existing_files) + 1  # 이어서 번호 시작

    # 3. 새 파일명 생성
    new_names = [f"AIHUB_ACTOR_{i:05}.wav" for i in range(start_idx, start_idx + len(all_files))]

    # 4. 멀티프로세싱 실행 (진행률 tqdm)
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(copy_file, f, new_names[idx]) for idx, f in enumerate(all_files)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="복사 중 ..."):
            result = future.result()
            if result is not True:
                print(result)
