# 데스크톱 관제 콘솔 (RipCurrentConsole)

이안류 모델과 사람 모델(ONNX)로 해안 영상을 분석하고, 이안류 구역 안에 사람이 들어가면 경보를 기록하고 FCM 푸시를 보내는 Windows 관제 프로그램입니다.

- **플랫폼:** Windows x64, .NET 8 (WPF)
- **추론:** ONNX Runtime + DirectML (GPU 가속, 실패 시 CPU)
- **영상 처리:** OpenCvSharp4
- **저장 / 알림:** SQLite, Firebase Admin SDK (FCM)

## 구조

```
RipCurrentConsole/
├── MainWindow.xaml(.cs)        메인 관제 화면, 영상 처리 루프, 침범 판정, 사고 세션 필터
├── YoloDetector.cs             ONNX Runtime YOLO 추론 (전처리 · NMS)
├── AlertDispatcher.cs          FCM 발송, 스냅샷 제공용 내장 HTTP 서버 (포트 8501)
├── DatabaseLogger.cs           SQLite 경보 이력 (emergency_logs 테이블)
├── LogViewerWindow.xaml(.cs)   경보 이력 조회, 스냅샷 보기
├── SettingsWindow.xaml(.cs)    임계값 · 필터 설정 화면
├── AppSettings.cs              config.json 저장 / 불러오기
└── Models/
    ├── rip_current.onnx        이안류 탐지 모델 (Git LFS)
    └── swimmer.onnx            사람 탐지 모델 (Git LFS)
```

## 처리 흐름

1. 동영상 파일을 불러와 관제를 시작합니다.
2. 3프레임마다 한 번 처리합니다. 이안류 추론은 그중 다시 3번에 1번만 실행하고 결과를 캐시합니다.
3. 사람 박스가 이안류 박스와 겹치면 위험 인원으로 판정해 화면에 표시합니다.
4. **3프레임 연속** 위험이면 사고 세션을 시작하고 스냅샷을 저장합니다.
   - 같은 구역은 `SameAreaSuppressSec` 동안 다시 캡처하지 않습니다.
   - 새 위치(`SpatialDistanceThreshold` 밖)에 사람이 나타나거나 위험 인원이 늘면 바로 다시 캡처합니다.
   - 위험이 `SessionClearDelaySec` 동안 없으면 세션을 끝냅니다.
5. 스냅샷과 경보 기록을 SQLite에 저장하고 FCM 토픽 `rip_current_alert`로 알림을 보냅니다.

## 설정 (`config.json`, 설정 화면에서 변경)

| 항목 | 기본값 | 설명 |
| --- | --- | --- |
| `RipConf` / `RipIou` | 0.25 / 0.70 | 이안류 모델 임계값 |
| `SwimmerConf` / `SwimmerIou` | 0.10 / 0.45 | 사람 모델 임계값 |
| `SameAreaSuppressSec` | 13.0 | 같은 구역 재캡처 억제 시간(초) |
| `SessionClearDelaySec` | 4.0 | 위험이 사라진 뒤 세션 종료까지 시간(초) |
| `SpatialDistanceThreshold` | 220 | 같은 사람으로 볼 거리(px) |

## 실행

1. Git LFS로 ONNX 모델을 받습니다: `git lfs install && git lfs pull`
2. Firebase 콘솔에서 받은 `serviceAccountKey.json`을 **실행 파일과 같은 폴더**에 넣습니다. 저장소에는 포함되어 있지 않습니다.
3. `AlertDispatcher.cs`의 `LAPTOP_IP`를 관제 PC의 Wi-Fi IP로 바꿉니다. 폰이 이 주소로 스냅샷 이미지를 받아갑니다.
4. 빌드 및 실행:

```bash
dotnet run --project RipCurrentConsole
```

배포용 단일 실행 파일:

```bash
dotnet publish RipCurrentConsole -c Release -r win-x64 --self-contained true -p:PublishSingleFile=true -p:IncludeNativeLibrariesForSelfExtract=true -o ./dist
```

`Models/` 폴더는 빌드할 때 출력 폴더로 자동 복사됩니다(`.csproj`의 `CopyToOutputDirectory`).

## 한계

- 실시간 RTSP 스트림이 아니라 동영상 파일을 입력으로 받습니다.
- 침범 판정은 박스끼리 겹치는지로 봅니다. 웹 콘솔은 이안류 폴리곤과 사람 발 위치로 판정합니다.
- 관제 PC IP와 위치 정보(해운대, `CCTV-E01`)가 코드에 고정되어 있습니다.
