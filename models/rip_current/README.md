# 🌊 이안류 탐지 모델 (Rip Current Detection)

해안 CCTV 영상에서 이안류(Rip Current) 구역을 탐지하는 YOLO11 모델의 데이터 정제·학습·평가 코드입니다.
원천 데이터 279,151장을 정제해 **Dataset v2 (11,460장, 7개 해변)** 를 만들고, 같은 데이터로 **YOLO11 s / m / x** 를 비교해 **YOLO11m** 을 최종 모델로 선정했습니다.

> 미지 해변 = 학습·검증에서 완전히 격리한 낙산·대천 400장. 처음 보는 해변에서의 실전 성능을 보기 위한 평가셋입니다.

---

## 📊 결과

### 이안류 탐지 모델 비교 (같은 Dataset v2)

| 모델 | 파라미터 | 검증 mAP50 | 미지 해변 mAP50 | 미지 해변 Recall | F1 |
| --- | :---: | :---: | :---: | :---: | :---: |
| YOLO11s | 9.4M | 0.7508 | 0.2336 | 0.3046 | 0.3909 |
| **YOLO11m ★** | 20.1M | **0.7526** | **0.2546** | **0.3664** | **0.4023** |
| YOLO11x | 56.9M | 0.7380 | 0.0880 | 0.0662 | 0.1197 |

- **최종 모델 YOLO11m:** 검증 mAP50 0.7526 · Precision 0.7200 · Recall 0.6975 / 미지 해변 mAP50 0.2546 · Recall 0.3664 · F1 0.4023
- 가장 큰 YOLO11x는 해안 배경을 외워 미지 해변에서 Recall 0.0662로 무너졌습니다. 모델 크기와 일반화 성능이 **역 U자 관계**임을 확인했고, 전 지표 1위인 11m을 배포 모델로 골랐습니다.

### 미지 해변 Recall 변화 (낙산·대천 400장)

| 모델 | 미지 해변 Recall |
| --- | :---: |
| 1차 YOLOv8m (무작위 분할) | 0.2031 |
| YOLO11s | 0.3046 |
| **YOLO11m** | **0.3664 (+80.4%)** |
| YOLO11x | 0.0662 |

---

## 🛠️ 문제와 해결

| 문제 | 원인 · 조치 | 결과 |
| --- | --- | --- |
| 검증 mAP50 0.7410인데 미지 해변 Recall 0.2031 (미탐 79.7%) | 1차 무작위 분할에서 같은 카메라·시간대 프레임이 Train/Val에 섞여 검증 점수가 부풀려짐 → **15초 간격 샘플링 + 날짜·시간·카메라 기준 GroupKFold 분할** | 미지 해변 Recall **0.3664** |
| 세그멘테이션 학습 붕괴 (mask mAP50 0.0042) | SAM으로 만든 마스크 라벨의 품질 문제 → **BBox 탐지로 전환** | 검증 mAP50 **0.7526** |
| 큰 모델의 일반화 실패 | 같은 데이터로 s·m·x 비교 → 11x는 배경 과적합 | 11m을 최종 모델로 선정 |

---

## 📂 데이터셋 (Dataset v2)

### 4단계 정제

원천 데이터 **279,151장 → 11,460장**

1. **15초 간격 샘플링:** CCTV 연속 프레임은 거의 같은 장면이라, 같은 카메라 기준 15초 간격으로만 남김
2. **GroupKFold 시공간 분할:** 날짜·시간·카메라를 그룹으로 묶어 같은 장면이 Train과 Val에 동시에 들어가지 않게 분할 (데이터 누수 방지)
3. **주간 1080p 이상 선별:** 야간·저해상도 영상 제외
4. **해외 해변 결합:** 국내 AI-Hub「이안류 CCTV 데이터」에 스페인 발레아레스 해변 2곳 데이터를 결합 (OBB 라벨 → HBB 변환)

### 데이터 분할

| Split | 구성 | 이미지 수 |
| --- | --- | :---: |
| **Train** | 국내 + 스페인 해변, GroupKFold 분할 | 9,810 |
| **Valid** | 국내 + 스페인 해변, GroupKFold 분할 | 1,250 |
| **Test (미지 해변)** | 낙산·대천 — 학습·검증에서 완전 격리 | 400 |

- 파도 포말에 의한 오탐을 막기 위해 이안류가 없는 **배경 이미지를 20%** 편성했습니다.

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
   - 동일 카메라 기준 15초 간격 샘플링, 주간 1080p 이상 선별
   - 날짜·시간·카메라 기준 GroupKFold로 Train/Val 분할, 낙산·대천은 미지 해변 Test로 격리

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