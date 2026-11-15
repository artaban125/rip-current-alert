from pathlib import Path
import torch
from ultralytics import YOLO

def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    print("=" * 65)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🚀 학습 가속 디바이스: {gpu_name} (VRAM: {vram:.1f} GB)")
    else:
        print("⚠️ 경고: GPU를 찾을 수 없습니다.")
    print("=" * 65)

    # 1. YAML 경로 바인딩
    CURRENT_DIR = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()
    DATA_YAML = str(CURRENT_DIR / "dataset_integrated_v2" / "data_integrated_v2.yaml")

    # 2. YOLO11m 모델 및 실험명 지정
    EXP_NAME = "yolo11m_integrated_v2_ep70_pat15"
    model = YOLO("yolo11m.pt")  # 자동으로 yolo11m.pt 다운로드 후 로드

    # 3. 70 에포크 본 학습 (조기 종료 patience=15)
    results = model.train(
        data=DATA_YAML,
        epochs=70,
        patience=15,
        imgsz=1024,
        batch=4,  # 핵심: 8 -> 4로 변경 (VRAM ~5.8GB 안착)
        device=device,
        workers=4,
        amp=True,
        name=EXP_NAME,
        exist_ok=True,
        optimizer="SGD",
        lr0=0.0025,  # Linear scaling (batch 8->4 감소에 맞춰 0.0025로 안정화)
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        close_mosaic=10,
        save=True,
        save_period=5,
        val=True,
        plots=True,
        verbose=True,
    )

    print("\n" + "=" * 65)
    print("🎉 YOLO11m 본 학습 완료 또는 수렴 종료!")
    print(f"📁 최적 가중치: runs/detect/{EXP_NAME}/weights/best.pt")
    print("=" * 65)

if __name__ == "__main__":
    main()