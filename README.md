# 🌊 이안류 긴급 알리미 (Rip Current Alert)

해안 CCTV 영상에서 **이안류 위험 구역**과 **입수자**를 동시에 탐지하고, 위험 구역 안에 사람이 들어가면 **관찰 · 경고 · 긴급** 3단계로 구분해 **현장 안전요원의 스마트폰으로 즉시 알리는** 관제 보조 시스템입니다.

> 4인 팀 프로젝트 · 10일 MVP · 2026.09
> 기존 이안류 감시는 "이안류가 발생했는가"만 알려 줍니다. 이 프로젝트는 **"이안류 구역 안에 사람이 있는가"** 까지 판단해, 안전요원이 넓은 화면을 육안으로 뒤지는 시간을 줄이는 것을 목표로 했습니다.

> ⚠️ 이 시스템의 출력은 **관제 보조 정보**입니다. 이안류 발생을 공식 확정하거나 구조를 자동 요청하지 않으며, 모든 판단은 현장 안전요원의 확인을 전제로 합니다.

<br>

## 동작 흐름

| 단계 | 하는 일 | 구현 위치 |
| --- | --- | --- |
| 1. 영상 입력 | CCTV 녹화 영상 업로드(웹) 또는 동영상 파일 로드(데스크톱) | `apps/web_console`, `apps/desktop_console` |
| 2. 이안류 구역 탐지 | YOLO11m 이안류 모델로 위험 수역 검출 | `models/rip_current` |
| 3. 입수자 탐지 | YOLO11m 사람 모델 (웹은 SAHI 768px 타일 슬라이싱으로 작은 사람 보강) | `models/person` |
| 4. 구역 침범 판정 | 사람 위치가 이안류 구역 안에 있는지 판정 → 관찰 / 경고 / 긴급 | 각 콘솔의 판정 로직 |
| 5. 알림 게이트 | 연속 판정 · 쿨다운 · 인원 증가 시 재발송으로 알림 남발 방지 | `apps/web_console/alert`, `apps/desktop_console` |
| 6. 푸시 알림 | 스냅샷 + 위험 등급 + 인원 수를 FCM 토픽(`rip_current_alert`)으로 발송 | Firebase Cloud Messaging |
| 7. 현장 확인 | 안드로이드 앱에서 알림 수신, 현장 사진 확대 확인 | `apps/android_alert` |

| 위험 등급 | 조건 |
| --- | --- |
| 🟢 관찰 (watch) | 이안류 의심 구역 없음 |
| 🟠 경고 (warn) | 이안류 구역은 있으나 구역 안에 사람 없음 |
| 🔴 긴급 (emergency) | 이안류 구역 안에 사람 감지 → 알림 발송 |

<br>

## 폴더 구조

```
rip-current-alert/
├── models/
│   ├── rip_current/        이안류 탐지 모델 학습 (YOLO11 s/m/x, AI-Hub 이안류 CCTV 데이터)
│   └── person/             사람(입수자) 탐지 모델 학습·파인튜닝 및 결과 (YOLO11m)
├── apps/
│   ├── desktop_console/    데스크톱 관제 콘솔 (.NET 8 WPF, ONNX Runtime + DirectML)
│   ├── web_console/        웹 관제 콘솔 (Streamlit, Ultralytics + SAHI)
│   │   └── alert/          FCM 발송 · 알림 게이트 · 이벤트 저장
│   └── android_alert/      안드로이드 알림 앱 (Kotlin, FCM 수신)
└── docs/
    └── 01_기획보고서.md     프로젝트 기획보고서 (최종본)
```

<br>

## 기술 스택

| 영역 | 기술 |
| --- | --- |
| 탐지 모델 | Ultralytics YOLO11 (s / m / x), SAHI |
| 학습 환경 | PyTorch, NVIDIA RTX 4080 SUPER · A100 |
| 데스크톱 콘솔 | C# .NET 8 WPF, ONNX Runtime (DirectML), OpenCvSharp4, SQLite, Firebase Admin |
| 웹 콘솔 | Python, Streamlit, OpenCV, SQLite, firebase-admin |
| 모바일 | Kotlin, Android, Firebase Cloud Messaging |
| 데이터 | AI-Hub 이안류 CCTV 데이터, Roboflow search-and-rescue |

<br>

## 실행

저장소에는 **학습 데이터, PyTorch 가중치(`*.pt`), Firebase 키 파일이 포함되어 있지 않습니다.** 실행 전에 직접 준비해야 합니다.

| 필요한 파일 | 위치 | 비고 |
| --- | --- | --- |
| `serviceAccountKey.json` | `apps/web_console/alert/`, 데스크톱 실행 폴더 | Firebase 콘솔 → 서비스 계정에서 발급 |
| `google-services.json` | `apps/android_alert/app/` | Firebase 콘솔에서 안드로이드 앱 등록 후 발급 |
| `best_yolo11m_integrated_v2_first.pt`, `best_swimmer_yolo11m_b8.pt` | `apps/web_console/` | 웹 콘솔용 이안류 / 사람 모델 가중치 |

### 모델 학습

```bash
# 이안류 모델 — 하드웨어에 맞는 스크립트 선택
cd models/rip_current
pip install ultralytics torch torchvision
python yolo11m-integrated_v2_ep70.py

# 사람 모델
cd models/person
pip install -r requirements.txt
python src/person_model.py            # 1차 학습
python src/person_model_finetune.py   # 2차 파인튜닝
```

### 데스크톱 관제 콘솔

ONNX 모델 2개(`Models/*.onnx`, 약 160MB)는 **Git LFS**로 관리됩니다. clone 전에 `git lfs install`을 실행해 두세요.

```bash
cd apps/desktop_console
dotnet run --project RipCurrentConsole
```

자세한 내용은 [`apps/desktop_console/README.md`](apps/desktop_console/README.md)를 참고하세요.

### 웹 관제 콘솔

```bash
cd apps/web_console
pip install -r requirements.txt
pip install ultralytics sahi torch opencv-python numpy   # 추론에 필요한 패키지
streamlit run app.py
```

### 안드로이드 알림 앱

1. `apps/android_alert`를 Android Studio로 엽니다.
2. `app/google-services.json`을 넣고 실기기에 설치합니다.
3. 웹 콘솔과 연동할 때는 `adb reverse tcp:8501 tcp:8501`로 폰이 PC의 Streamlit 서버(스냅샷 이미지)에 접근하도록 합니다.

<br>

## 통합 코드의 원본

이 저장소는 팀원들이 각자 관리하던 저장소를 하나로 모은 것입니다. 프로젝트 당시에는 Git 사용을 염두에 두지 않아 커밋 기록이 실제 기여를 잘 보여 주지 않으므로, 기록은 가져오지 않고 파일만 옮겼습니다. 기여 내용은 아래 [팀원별 작업](#팀원별-작업)에 정리했습니다.

| 이 저장소 | 원본 저장소 | 원본 경로 |
| --- | --- | --- |
| `models/person/` | [Miita212/CV_Project_person_detect](https://github.com/Miita212/CV_Project_person_detect) | `/` |
| `apps/web_console/` | [gonggong77/team_beavers](https://github.com/gonggong77/team_beavers) | `rip_service_console/` |
| `apps/android_alert/` | [gonggong77/team_beavers](https://github.com/gonggong77/team_beavers) | `rip_current_app/rip_current_app/AndroidStudioProjects/RipCurrentAlert/` |
| `docs/01_기획보고서.md` | 팀 Notion 운영 보드 | 기획서 최종 |

옮기면서 바꾼 점:
- 개인 IDE 설정(`*.csproj.user`, `*.pubxml.user`, `.idea/`, `.vs/`)과 Firebase 설정 파일(`google-services.json`)은 제외했습니다.
- 각 폴더 README의 경로와 링크를 이 저장소 구조에 맞게 고쳤습니다. 코드는 원본 그대로입니다.

<br>

## 팀원별 작업

| 담당 영역 | 주요 결과물 | 서비스에 쓰인 곳 |
| --- | --- | --- |
| 이안류 탐지 모델 · 데스크톱 관제 콘솔 (artaban125) | YOLO11 s/m/x 이안류 모델, .NET 8 WPF 관제 콘솔 | 두 콘솔의 이안류 구역 탐지, 데스크톱 관제 전체 |
| 사람(입수자) 탐지 모델 | YOLO11m 사람 모델 1차 학습 + 작은 사람 파인튜닝 | 두 콘솔의 입수자 탐지 (`swimmer` 모델) |
| 웹 관제 콘솔 | Streamlit 영상 분석 화면, 듀얼 모델 추론 파이프라인 | 시연용 웹 관제 화면 |
| 알림 서비스 · 안드로이드 앱 | FCM 발송 모듈, 알림 게이트, Android 수신 앱 | 긴급 알림 발송·수신 전체 |

### 이안류 탐지 모델 · 데스크톱 관제 콘솔 — artaban125

**`models/rip_current/` — 이안류 탐지 모델**
- **데이터 정제 (`dataset.ipynb`)**
  - AI-Hub「이안류 CCTV 데이터」의 파일명 메타데이터(`[해수욕장]_[카메라]_[YYYYMMDD]_[HHMMSS]`)를 파싱해 지점·시간대별 분포 분석
  - JSON 라벨 좌표를 YOLO 포맷(`cls x_center y_center w h`)으로 정규화 변환
  - 같은 카메라 기준 **15초 이내 연사 프레임 제거**로 중복 이미지 정리
- **도메인 격리 분할 (데이터 누수 방지)**
  - Train / Valid: 해운대·송정·중문을 **날짜 단위**로 분할 (9,200 / 1,200장)
  - Test: 낙산·대천을 **학습에 쓰지 않은 지점**으로 완전 격리 (600장)
  - Positive(이안류) 대 Negative(배경) 비율을 맞춰 오탐 제어
- **YOLO11 스케일별 학습 스크립트**
  - `yolo11s`: 8GB VRAM 로컬 학습 (SGD, batch 8, AMP)
  - `yolo11m`: batch 4 + 선형 학습률 스케일링(`lr0=0.0025`)으로 약 5.8GB VRAM에서 1024 해상도 학습
  - `yolo11x`: A100 전용 (TF32 / bfloat16, AdamW + cosine LR, mixup 0.15)

**`apps/desktop_console/` — 데스크톱 관제 콘솔 (C# .NET 8 WPF)**
- **ONNX 추론 엔진 (`YoloDetector.cs`)**: ONNX Runtime + **DirectML GPU 가속**, 모델 입력 크기 자동 감지, GPU 실패 시 CPU로 대체하고 원인을 팝업으로 표시
- **실시간 처리 루프 (`MainWindow.xaml.cs`)**
  - 3프레임마다 처리하고, 이안류 추론은 그중 다시 3번에 1번만 돌린 뒤 결과를 캐시해 속도 확보
  - 사람 박스와 이안류 박스의 **교차 영역으로 구역 침범 판정**, 화면에 RIP / DANGER 오버레이
  - 상단 알림바로 관찰·경고·긴급 3단계 표시
- **사고 세션 필터링**: **3프레임 연속** 위험일 때만 경보, 같은 구역은 일정 시간 재캡처 억제, **새 위치 또는 인원 증가 시 즉시 재캡처**, 위험이 사라지면 세션 종료
- **알림 발송 (`AlertDispatcher.cs`)**: Firebase Admin SDK로 FCM 토픽 발송, 스냅샷을 폰에서 받을 수 있도록 **내장 HTTP 서버**로 제공
- **경보 이력 (`DatabaseLogger.cs`, `LogViewerWindow`)**: SQLite에 경보 기록, 발송 상태·확인 여부 관리, 스냅샷 보기
- **설정 화면 (`SettingsWindow`, `AppSettings.cs`)**: 모델별 conf/IoU, 재캡처 억제 시간, 세션 종료 지연, 동일인 판정 반경을 `config.json`으로 저장

### 사람(입수자) 탐지 모델

**`models/person/`** — 자세한 과정과 수치는 [`models/person/README.md`](models/person/README.md)에 있습니다.
- **데이터 정제**: Roboflow 원본 4,955장 → 유사 프레임 제거(64×64 축소 MSE 비교) → 배경 비율 조정 → **1,028장** (사람 878 · 배경 150)
- **1차 학습**: YOLO11m, 1024px, 모자이크 증강 끔 → mAP50 0.827. 그러나 실제 CCTV의 몇 픽셀짜리 사람을 거의 못 잡는 문제 발견
- **2차 파인튜닝**: 작은 사람 위주 데이터 재구성, 해상도 1280, 객체를 작게 만드는 증강(mosaic·mixup·erasing) 끔, cls 손실 가중치 상향, 낮은 학습률 → mAP50 **0.907**
- **서비스 적용 제안**: 넓은 화면의 작은 사람 대응을 위해 SAHI 타일 슬라이싱을 함께 쓰는 방식으로 웹 콘솔에 적용
- **프로젝트 이후 추가 검증 (2026.11)**: 학습에 쓰지 않은 서프캠 데이터 2,413장으로 재평가해 실제 성능(mAP50 0.41 → 0.50)과 파인튜닝 효과를 확인

### 웹 관제 콘솔

**`apps/web_console/`** (`app.py`, `views/`, `core/`)
- **Streamlit 관제 화면**: 사이드바 영상 업로드 → 분석 시작, 실시간 오버레이 영상, 위험 등급 상·하단 바, 캡처 갤러리, 우측 경보 카드, 실시간 로그, 경보 "확인 완료" 처리
- **듀얼 모델 추론 파이프라인 (`core/pipeline.py`)**
  - 이안류: YOLO11m(640px, conf 0.25) 검출 박스 → 구역 폴리곤으로 변환
  - 사람: **SAHI 768px 타일(overlap 18%)** + YOLO11m, conf 0.10
  - GPU FP16, `torch.inference_mode()`로 추론 가속, 스냅샷 인코딩과 발송은 백그라운드 스레드로 넘겨 분석이 멈추지 않게 함
- **판정 규칙 (`core/rules.py`)**: 사람 박스 **발 위치(하단 중앙)** 가 이안류 폴리곤 안(경계 15px 여유)에 있는지로 판정
- **렌더링 (`core/render.py`)**: 이안류 반투명 채우기, 사람 안전/위험 색 구분, 등급별 색상과 설명 문구

### 알림 서비스 · 안드로이드 앱

**`apps/web_console/alert/` — 알림 발송 모듈**
- **FCM 발송 (`fcm_alert.py`)**: 서비스 계정 키 경로 자동 탐색(환경변수 `FIREBASE_CREDENTIALS` 지원), 스냅샷을 Streamlit 정적 서빙(`/app/static/`)으로 제공해 별도 스토리지 없이 이미지 URL 전달
- **알림 게이트 (`gate.py`)**: 연속 판정 횟수 + 쿨다운으로 알림 남발 방지, **위험 인원이 늘면 즉시 재발송**
- **알림 어댑터 (`notifier.py`, `sender.py`)**: 웹에서는 `notifier.send(event)` 한 줄로 발송, 발송은 백그라운드에서 처리
- **이벤트 저장 (`store.py`)**: 발송 성공 여부와 관계없이 모든 이벤트를 SQLite에 기록해 재전송·디버깅 가능
- **발송 테스트 (`send_test.py`)**: 화면 없이 CLI로 알림 발송 확인

**`apps/android_alert/` — 안드로이드 알림 앱 (Kotlin)**
- **FCM 수신 (`MyFirebaseMessagingService.kt`)**: `rip_current_alert` 토픽 구독, 높은 중요도 알림 채널과 진동, 위험 등급별 알림 색상
- **알림 화면 (`MainActivity.kt`)**: 구역 · 카메라 ID, 구역 내 인원 / 전체 인원 / 이안류 구역 수, 감지 시각, 위험 등급 색상 표시, **현장 스냅샷을 받아와 핀치 줌·이동으로 확인**
- **앱 내 이벤트 전달 (`AlertBus.kt`)**: 앱이 켜져 있을 때 받은 알림을 화면에 바로 반영

<br>

## 결과와 한계

- 이안류 구역과 입수자를 함께 탐지하고, 구역 안 인원이 생기면 **수 초 안에 스마트폰으로 현장 사진이 포함된 알림**이 도착하는 흐름을 웹·데스크톱 두 가지 콘솔로 구현했습니다.
- 사람 모델은 학습 데이터 기준 점수(mAP50 0.91)와 처음 보는 데이터 점수(0.50)의 차이가 컸습니다. 학습과 분리된 평가셋을 처음부터 만드는 것이 다음 과제입니다.
- 실제 CCTV 실시간 스트림 연동, 해양 관측 데이터(이안류 지수·파고·풍속) 결합, 다중 카메라 대시보드는 향후 확장 범위로 남겼습니다.

자세한 기획 배경과 요구사항은 [`docs/01_기획보고서.md`](docs/01_기획보고서.md)를 참고하세요.
