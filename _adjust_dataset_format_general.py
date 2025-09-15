import os
import json
from pydub import AudioSegment

def split_audio_by_json(audio_root, json_root, output_root):
    os.makedirs(output_root, exist_ok=True)

    # 모든 wav 파일 경로 사전 구축 {basename: full_path}
    wav_map = {}
    for root, _, files in os.walk(audio_root):
        for f in files:
            if f.endswith(".wav"):
                base = os.path.splitext(f)[0]
                wav_map[base] = os.path.join(root, f)

    # JSON 파일 순회
    for root, _, files in os.walk(json_root):
        for f in files:
            if not f.endswith(".json"):
                continue

            base = os.path.splitext(f)[0]
            json_path = os.path.join(root, f)

            if base not in wav_map:
                print(f"[경고] 매칭되는 wav 없음: {base}")
                continue

            audio_path = wav_map[base]

            # 오디오 로드
            try:
                audio = AudioSegment.from_wav(audio_path)
            except Exception as e:
                print(f"[에러] 오디오 로드 실패: {audio_path} ({e})")
                continue

            # JSON 로드
            try:
                with open(json_path, "r", encoding="utf-8") as jf:
                    data = json.load(jf)
            except Exception as e:
                print(f"[에러] JSON 로드 실패: {json_path} ({e})")
                continue

            # Conversation 순회
            for idx, seg in enumerate(data.get("Conversation", [])):
                try:
                    start_ms = int(float(seg["StartTime"]) * 1000)
                    end_ms = int(float(seg["EndTime"]) * 1000)
                    emotion = seg.get("SpeakerEmotionTarget", "Unknown")

                    segment = audio[start_ms:end_ms]

                    # 감정별 폴더
                    emotion_dir = os.path.join(output_root, emotion)
                    os.makedirs(emotion_dir, exist_ok=True)

                    out_file = f"{base}_{idx}.wav"
                    out_path = os.path.join(emotion_dir, out_file)

                    segment.export(out_path, format="wav")
                    print(f"[저장 완료] {out_path}")

                except Exception as e:
                    print(f"[에러] 세그먼트 처리 실패 ({base}_{idx}): {e}")


if __name__ == "__main__":
    # 경로 설정 (필요에 맞게 수정)
    AUDIO_ROOT = r"E:\03_산학연_음성감정인식\01_감정이 태깅된 자유대화 (성인)\01-1.정식개방데이터\Training\01.원천데이터"
    JSON_ROOT  = r"E:\03_산학연_음성감정인식\01_감정이 태깅된 자유대화 (성인)\01-1.정식개방데이터\Training\02.라벨링데이터"
    OUTPUT_ROOT = r"E:\03_datasets\datasets_aihub_general"

    split_audio_by_json(AUDIO_ROOT, JSON_ROOT, OUTPUT_ROOT)
