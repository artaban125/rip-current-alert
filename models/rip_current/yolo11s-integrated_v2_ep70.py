from pathlib import Path
import torch
from ultralytics import YOLO


def main():
    # 1. GPU 디바이스 설정 및 확인
    device = 0 if torch.cuda.is_available() else "cpu"
    print("=" * 65)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🚀 학습 가속 디바이스: {gpu_name} (VRAM: {vram:.1f} GB)")
    else:
        print("⚠️ 경고: GPU를 찾을 수 없습니다.")
    print("=" * 65)

    # 2. YAML 데이터셋 경로 바인딩
    CURRENT_DIR = (
        Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()
    )
    DATA_YAML = str(
        CURRENT_DIR / "dataset_integrated_v2" / "data_integrated_v2.yaml"
    )

    print(f"📌 데이터셋 YAML 경로: {DATA_YAML}")
    assert Path(DATA_YAML).exists(), f"❌ YAML 파일을 찾을 수 없습니다: {DATA_YAML}"

    # 3. 모델 및 실험명 설정 (project 인자 없이 단일 runs/detect 생성)
    EXP_NAME = "yolo11s_integrated_v2_ep70_pat15"
    model = YOLO("yolo11s.pt")

    # 4. 70 에포크 + 15 조기 종료 본 학습 시작
    results = model.train(
        data=DATA_YAML,
        epochs=70,  # 최대 탐색 상한
        patience=15,  # 15 에포크 연속 mAP50 개선 없으면 자동 조기 종료
        imgsz=1024,
        batch=8,  # 8GB VRAM 최적 배치 크기
        device=device,
        workers=4,  # Windows 환경 최적 워커 수
        amp=True,  # FP16 자동 혼합 정밀도 (속도 2배 향상 및 VRAM 절약)
        name=EXP_NAME,
        exist_ok=True,
        # 일반화 최적 옵티마이저 (SGD 표준 세팅)
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        # 증강 기법 제어
        close_mosaic=10,  # 마지막 10 에포크 전부터 모자이크 OFF (실사 정밀 수렴)
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        # 저장 및 모니터링
        save=True,
        save_period=5,
        val=True,
        plots=True,
        verbose=True,
    )

    print("\n" + "=" * 65)
    print("🎉 YOLO11s 본 학습 완료 또는 조기 종료 수렴!")
    print(f"📁 최적 가중치: runs/detect/{EXP_NAME}/weights/best.pt")
    print(f"📁 최종 로그 및 차트: runs/detect/{EXP_NAME}")
    print("=" * 65)


if __name__ == "__main__":
    main()