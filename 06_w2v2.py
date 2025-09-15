#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Wav2Vec2 감정 분류 (safetensors 전용)
- 입력 CSV: split_dir/{train,val,test}.csv with [filepath, label, label_id]
- mode:
  * train : 학습 + 검증 수행 후 저장
  * eval  : 저장된 모델 로드 후 테스트셋 평가(Accuracy/Report/CM)
"""

import os
import argparse
import warnings
from glob import glob
import unicodedata

import numpy as np
import pandas as pd
import librosa

import torch
from torch.utils.data import DataLoader

from datasets import Dataset as HFDataset, DatasetDict

import transformers as _tf
from transformers import (
    AutoProcessor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_utils import set_seed
from packaging import version
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm


# ----------------------------
# 경로 유틸
# ----------------------------
def norm_abs_path(p: str) -> str:
    p = str(p).strip()
    p = os.path.expanduser(p)
    p = os.path.abspath(p)
    return unicodedata.normalize("NFC", p)


# ----------------------------
# 데이터 로딩
# ----------------------------
def load_split_csvs(split_dir: str):
    split_dir = norm_abs_path(split_dir)

    def _read(name):
        p = os.path.join(split_dir, f"{name}.csv")
        assert os.path.exists(p), f"missing {name}.csv in {split_dir}"
        df = pd.read_csv(p)
        df["filepath"] = df["filepath"].astype(str).map(norm_abs_path)
        return df

    tr = _read("train")
    va = _read("val")
    te = _read("test")
    return tr, va, te


def build_label_maps(df_train: pd.DataFrame):
    labels = sorted(df_train["label"].unique().tolist())
    ids = sorted(df_train["label_id"].unique().tolist())
    assert len(labels) == len(ids), "label과 label_id 매핑 불일치 가능성"
    tmp = df_train.drop_duplicates(subset=["label", "label_id"]).sort_values("label_id")
    id2label = {int(r.label_id): str(r.label) for _, r in tmp.iterrows()}
    label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id


def filter_missing_files(df: pd.DataFrame, name: str) -> pd.DataFrame:
    exists_mask = df["filepath"].map(os.path.exists)
    missing = (~exists_mask).sum()
    if missing > 0:
        print(f"[WARN] {name}: missing files = {missing} (행 제거)")
    return df[exists_mask].reset_index(drop=True)


# ----------------------------
# HF Dataset 변환 + 전처리
# ----------------------------
def to_hf_dataset(df: pd.DataFrame) -> HFDataset:
    return HFDataset.from_pandas(df[["filepath", "label_id"]], preserve_index=False)


def make_preprocess_fn(processor, target_sr=16000, max_duration_s=None):
    max_len = int(target_sr * max_duration_s) if max_duration_s else None

    def _fn(batch):
        wav_path = batch["filepath"]
        try:
            y, _ = librosa.load(wav_path, sr=target_sr, mono=True)
        except Exception as e:
            raise RuntimeError(f"오디오 로드 실패: {wav_path} ({e})")
        if max_len and len(y) > max_len:
            y = y[:max_len]
        out = processor(y, sampling_rate=target_sr, return_tensors=None)
        batch["input_values"] = out["input_values"][0]
        batch["labels"] = int(batch["label_id"])
        return batch

    return _fn


# ----------------------------
# Collator (padding)
# ----------------------------
class CollatorForAudioClassification:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        inputs = [f["input_values"] for f in features]
        labels = torch.tensor([int(f["labels"]) for f in features], dtype=torch.long)
        batch = self.processor.pad({"input_values": inputs}, padding=True, return_tensors="pt")
        batch["labels"] = labels
        return batch


# ----------------------------
# 모델 로더 (safetensors only)
# ----------------------------
def load_model_safetensors_only(model_id, num_labels, id2label, label2id,
                                problem_type="single_label_classification"):
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        problem_type=problem_type,
        use_safetensors=True,
        ignore_mismatched_sizes=True,
    )
    return model


def forbid_bin_checkpoints(resume_dir: str | None):
    if not resume_dir:
        return None
    resume_dir = resume_dir.strip()
    if not resume_dir:
        return None
    resume_dir = norm_abs_path(resume_dir)
    if not os.path.isdir(resume_dir):
        return None

    sub_ckpts = [d for d in glob(os.path.join(resume_dir, "checkpoint-*")) if os.path.isdir(d)]
    if sub_ckpts:
        sub_ckpts.sort(key=os.path.getmtime, reverse=True)
        resume_dir = sub_ckpts[0]

    has_model = any(
        os.path.exists(os.path.join(resume_dir, f))
        for f in ["pytorch_model.safetensors", "model.safetensors",
                  "pytorch_model.bin", "model.bin"]
    )
    if not has_model:
        return None

    bins = glob(os.path.join(resume_dir, "**", "*.bin"), recursive=True)
    if bins:
        raise RuntimeError(
            "이 환경에서는 .bin 체크포인트를 로드할 수 없습니다. safetensors만 사용하세요."
        )
    return resume_dir


# ----------------------------
# 검증 메트릭(Trainer용)
# ----------------------------
def make_compute_metrics(num_labels: int):
    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        p, r, f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )
        return {
            "accuracy": float(acc),
            "macro_f1": float(f1),
            "macro_precision": float(p),
            "macro_recall": float(r),
        }
    return _compute_metrics


# ----------------------------
# 테스트: PyTorch 스타일 평가
# ----------------------------
def evaluate_model_pt(trainer, dataset, class_names=None, device="cuda"):
    model = trainer.model.to(device)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=trainer.args.per_device_eval_batch_size,
        collate_fn=trainer.data_collator,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating (PT)"):
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

    all_preds = np.asarray(all_preds)
    all_labels = np.asarray(all_labels)

    acc = float((all_preds == all_labels).mean())
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    return {"accuracy": acc, "y_true": all_labels, "y_pred": all_preds}


# ----------------------------
# 메인
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="train", choices=["train", "eval"],
                    help="train: 학습+검증만 수행, eval: 저장된 모델로 테스트셋 평가만 수행")
    ap.add_argument("--split_dir", type=str, default="./dataset_splits", help="CSV 폴더")
    ap.add_argument("--output_dir", type=str, default="./output", help="모델/프로세서 저장 또는 로드 경로")
    ap.add_argument("--model_id", type=str, default="facebook/wav2vec2-base")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--train_bs", type=int, default=8)
    ap.add_argument("--eval_bs", type=int, default=8)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--max_duration_s", type=float, default=None)
    ap.add_argument("--resume_from", type=str, default=None)
    ap.add_argument("--freeze_feature_extractor", action="store_true")
    ap.add_argument("--num_proc", type=int, default=1)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--max_steps", type=int, default=None)
    args = ap.parse_args()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

    set_seed(args.seed)

    # CSV 로드 + 결측 파일 필터
    tr, va, te = load_split_csvs(args.split_dir)
    tr = filter_missing_files(tr, "train")
    va = filter_missing_files(va, "val")
    te = filter_missing_files(te, "test")

    id2label, label2id = build_label_maps(tr)
    num_labels = len(id2label)

    # Processor
    # train 모드: 허브/모델 ID에서 로드
    # eval  모드: 저장된 디렉토리(output_dir)에서 로드 권장(동일 전처리)
    if args.mode == "eval" and os.path.isdir(norm_abs_path(args.output_dir)):
        processor = AutoProcessor.from_pretrained(args.output_dir)
    else:
        processor = AutoProcessor.from_pretrained(args.model_id)

    # Dataset + 전처리
    ds = DatasetDict({
        "train": to_hf_dataset(tr),
        "validation": to_hf_dataset(va),
        "test": to_hf_dataset(te),
    })
    preprocess_fn = make_preprocess_fn(processor, target_sr=16000, max_duration_s=args.max_duration_s)
    ds = ds.map(preprocess_fn, num_proc=max(1, args.num_proc))
    ds = ds.select_columns(["input_values", "labels"])

    # Collator
    collator = CollatorForAudioClassification(processor)

    # TrainingArguments (train일 때만 사용됨)
    _use_eval_strategy = version.parse(_tf.__version__) >= version.parse("4.46")
    common_kwargs = dict(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        num_train_epochs=args.epochs,
        save_strategy="epoch",
        save_total_limit=3,
        logging_steps=50,
        fp16=bool(args.fp16),
        bf16=False,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_safetensors=True,
        gradient_checkpointing=bool(args.gradient_checkpointing),
        max_steps=args.max_steps,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=False,
    )
    training_args = (
        TrainingArguments(eval_strategy="epoch", **common_kwargs)
        if _use_eval_strategy
        else TrainingArguments(evaluation_strategy="epoch", **common_kwargs)
    )

    # 모델 로드
    if args.mode == "eval":
        # 저장된 모델에서 로드(동일 processor 기준)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            args.output_dir, use_safetensors=True
        )
    else:
        model = load_model_safetensors_only(
            model_id=args.model_id,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )
        if args.freeze_feature_extractor and hasattr(model, "wav2vec2") and hasattr(model.wav2vec2, "feature_extractor"):
            for p in model.wav2vec2.feature_extractor.parameters():
                p.requires_grad = False
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

    # Trainer (processing_class로 경고 제거)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"] if args.mode == "train" else None,
        eval_dataset=ds["validation"] if args.mode == "train" else None,
        processing_class=processor,
        data_collator=collator,
        compute_metrics=make_compute_metrics(num_labels),
    )

    # 모드 분기
    if args.mode == "train":
        resume_dir = forbid_bin_checkpoints(args.resume_from)
        if resume_dir:
            print(f"[INFO] Resuming from: {resume_dir}")
            trainer.train(resume_from_checkpoint=resume_dir)
        else:
            print("[INFO] Fresh training (no resume).")
            trainer.train()

        val_metrics = trainer.evaluate(ds["validation"])
        print("=== Val metrics (Trainer) ===")
        for k, v in val_metrics.items():
            print(f"{k}: {v}")

        trainer.save_model()
        processor.save_pretrained(args.output_dir)

        # 테스트는 train 모드에서 자동으로 하지 않음(원하면 아래 한 줄 추가)
        # evaluate_model_pt(trainer, ds["test"], class_names=[id2label[i] for i in range(num_labels)])
    else:  # eval
        class_names = [id2label[i] for i in range(num_labels)]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _ = evaluate_model_pt(trainer, ds["test"], class_names=class_names, device=device)


if __name__ == "__main__":
    main()
