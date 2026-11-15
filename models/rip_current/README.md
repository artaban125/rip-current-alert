# 🌊 AI-based Rip Current Detection System (이안류 탐지 시스템)

해안가 CCTV 영상을 활용하여 실시간으로 이안류(Rip Current)를 탐지하고 해수욕객 안전사고를 예방하기 위한 딥러닝 객체 검출 프로젝트입니다.  
원천 데이터 분석, 시간적 중복 제거 및 지점/일자 기반 도메인 격리 파이프라인을 거쳐 **YOLO11 (s / m / x)** 모델군에 대한 최적화 학습을 수행합니다.

---

## 📌 주요 특징 (Key Highlights)

1. **체계적인 데이터 전처리 및 정제 파이프라인**
   - **초 단위 연사 중복 제거**: CCTV 특성상 프레임 간 유사도가 매우 높으므로 동일 지점 기준 최소 15초 간격 필터링 적용
   - **엄격한 도메인 격리 (Zero Data Leakage)**:
     - `Train / Valid`: 해운대(HD), 송정(SJ), 중문(JM) 데이터를 날짜(Date) 단위로 분할하여 과적합 방지
     - `Test`: 낙산(NS), 대천(DC) 지점을 미학습 평가 도메인으로 완전 격리하여 실제 환경에 대한 일반화 성능 검증
   - **배경 데이터(Negative Samples) 통합**: 미탐/오탐 제어를 위해 Positive(이안류) 대 Negative(배경) 비율을 체계적으로 배분

2. **스케일별 YOLO11 모델 튜닝 및 하드웨어 가속**
   - **YOLO11s**: 경량 엣지 환경 대응 (Batch 8, SGD, Mosaic 1.0)
   - **YOLO11m**: 균형 모델 (Linear Learning Rate Scaling, Batch 4 안정화)
   - **YOLO11x**: 고성능 A100 GPU 최적화 (TF32 & bfloat16 활성화, AdamW, Cosine LR, Mixup 증강)

---

## 📂 데이터셋 구조 및 분할 전략

AI Hub '이안류 CCTV 데이터' 원천 데이터셋을 정제하여 `dataset_integrated_v2` 형태로 재구성하였습니다.

### 데이터 분할 통계

| Split | 지점 (Beach Domain) | Positive (이안류) | Negative (배경) | Total |
| :--- | :--- | :---: | :---: | :---: |
| **Train** | 해운대(HD), 송정(SJ), 중문(JM) (일자 분할) | 8,000 | 1,200 | 9,200 |
| **Valid** | 해운대(HD), 송정(SJ), 중문(JM) (일자 분할) | 1,000 | 200 | 1,200 |
| **Test** | 낙산(NS), 대천(DC) **(완전 도메인 격리)** | 500 | 100 | 600 |

### 디렉토리 구조

```text
dataset_integrated_v2/
├── data_integrated_v2.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

`data_integrated_v2.yaml` 설정 예시:
```yaml
path: ./dataset_integrated_v2
train: train/images
val: valid/images
test: test/images

names:
  0: rip_current
```

---

## 🛠️ 프로젝트 구조

```text
├── dataset.ipynb                     # 메타데이터 분석, 1차 YOLO 포맷 변환, 2차 정제 파이프라인
├── yolo11s-integrated_v2_ep70.py     # YOLO11-Small 학습 스크립트 (SGD, Batch 8)
├── yolo11m-integrated_v2_ep70.py     # YOLO11-Medium 학습 스크립트 (VRAM 6GB 최적화)
├── yolo11x-integrated_v2_ep70.py     # YOLO11-XLarge A100 전용 학습 스크립트 (TF32/BF16, AdamW)
└── README.md
```

---

## 🚀 시작하기

### 1. 환경 구성

```bash
git clone https://github.com/artaban125/rip-current-alert.git
cd rip-current-alert/models/rip_current

pip install ultralytics torch torchvision
```

### 2. 데이터 전처리 (`dataset.ipynb`)

1. **원천 데이터 분석**: 파일명 메타데이터(`[해수욕장]_[카메라]_[YYYYMMDD]_[HHMMSS]`)를 파싱하여 지점 및 시간대별 분포 분석
2. **1차 분리 및 포맷 변환**: JSON 라벨링 좌표를 YOLO Bounding Box 포맷(`cls x_center y_center w h`)으로 정규화 변환
3. **2차 정제 및 분할**:
   - 동일 카메라 기준 15초 이내 연사 프레임 필터링
   - 낙산/대천 도메인 격리 및 날짜 기반 Train/Val 분할 수행

### 3. 모델 학습

사용 가능한 하드웨어 스펙에 맞춰 스크립트를 실행합니다.

#### YOLO11s (로컬 / 8GB VRAM)
```bash
python yolo11s-integrated_v2_ep70.py
```
- 이미지 크기: `1024`
- 배치 크기: `8`
- 옵티마이저: `SGD` (`lr0=0.01`, `momentum=0.937`)
- 주요 기법: `patience=15`, `close_mosaic=10`, `AMP(FP16)`

#### YOLO11m (VRAM 절약형)
```bash
python yolo11m-integrated_v2_ep70.py
```
- 배치 크기를 `4`로 낮추고 선형 스케일링 규칙 적용 (`lr0=0.0025`)
- 약 5.8GB 수준의 VRAM으로 안정적 1024 해상도 학습 가능

#### YOLO11x (A100 GPU 대용량 학습)
```bash
python yolo11x-integrated_v2_ep70.py
```
- `TF32` 및 `bfloat16` 텐서 코어 가속 활성화
- 옵티마이저: `AdamW` (`cos_lr=True`)
- 증강: `Mosaic=1.0`, `Mixup=0.15`, `Degrees=10.0`
- 배치 크기: `32`

---

## 📊 학습 모니터링 및 결과

학습 실행 시 `runs/detect/` 하위에 로그, 손실 곡선 및 Confusion Matrix, PR Curve가 자동 생성됩니다.

- **Checkpoints**: `runs/detect/<EXP_NAME>/weights/best.pt`
- **Validation**:
  ```python
  from ultralytics import YOLO

  model = YOLO("runs/detect/yolo11x_a100_integrated_v2/weights/best.pt")
  metrics = model.val(data="dataset_integrated_v2/data_integrated_v2.yaml", split="test")
  print(f"Test mAP50-95: {metrics.box.map}")
  ```

---

## 📄 License
This project is licensed under the Apache 2.0 License.