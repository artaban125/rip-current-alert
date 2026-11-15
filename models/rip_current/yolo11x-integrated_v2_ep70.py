import os
import torch
from ultralytics import YOLO

def train_yolo11x_a100():
    # 1. A100 하드웨어 가속 설정
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 2. 데이터셋 경로 (data.yaml)
    data_yaml_path = "/content/drive/MyDrive/dataset_integrated_v2/data.yaml"

    # 3. 모델 가중치 로드 (Detection: yolo11x.pt / Segmentation: yolo11x-seg.pt)
    model = YOLO("yolo11x.pt")

    # 4. A100 맞춤 파라미터 학습 실행
    results = model.train(
        data=data_yaml_path,
        epochs=70,
        batch=32,            # A100 VRAM을 활용한 배치 크기
        imgsz=1024,
        device=0,
        workers=8,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        amp=True,
        bfloat16=True,       # A100 텐서코어 전용 bfloat16
        patience=25,
        save=True,
        save_period=10,

        # 해양/이안류 데이터 증강 파라미터
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        close_mosaic=10,     # 마지막 10 에포크는 모자이크 해제

        project="/content/drive/MyDrive/runs/train_dataset_v2",
        name="yolo11x_a100_integrated_v2",
        exist_ok=True,
        plots=True,
        verbose=True
    )

    # 5. 검증 평가
    metrics = model.val(data=data_yaml_path, split="val")
    print(f"Validation mAP50-95: {metrics.box.map}")

if __name__ == "__main__":
    train_yolo11x_a100()