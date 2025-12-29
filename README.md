## 프로젝트 구조
```
k-league/
├── configs/
│   └── config.yaml          # OmegaConf 설정 파일 (모델, 학습, Feature 등)
├── src/
│   ├── data/
│   │   ├── dataset.py       # KLeagueDataset, KLeagueTestDataset
│   │   └── datamodule.py    # PyTorch Lightning DataModule
│   ├── models/
│   │   ├── transformer.py   # Transformer 인코더 모델
│   │   └── lightning_module.py  # Lightning 학습 모듈
│   └── utils/
│       ├── features.py      # Feature 추출기 (config로 동적 선택)
│       └── metrics.py       # 유클리드 거리 평가 지표
├── scripts/
│   ├── setup.sh            # 환경 설정 스크립트
│   ├── run_train.sh        # 학습 실행 스크립트
│   └── run_inference.sh    # 추론 실행 스크립트
├── train.py                # 학습 진입점 (MLflow + Early Stopping)
├── inference.py            # 추론 진입점 (자동 모델 로드)
├── requirements.txt        # 패키지 의존성
└── .gitignore
```
## 주요 기능
Transformer 모델: 시퀀스 트랜스포머 인코더로 패스 도착 좌표 예측
동적 Feature 선택: configs/config.yaml에서 사용할 feature 그룹 선택 가능
좌표 정규화: 학습 시 [0,1] 정규화, 추론 시 원래 스케일(105x68)로 복원
MLflow 실험 추적: 하이퍼파라미터, 메트릭, 모델 자동 로깅
Early Stopping: validation loss 기준 조기 종료
Best 모델 자동 로드: checkpoints/best_model_path.txt에서 자동 읽기

## 사용 방법
1. 환경 설정
```bash
bash scripts/setup.sh
source .venv/bin/activate
```
2. 학습
```bash
python train.py --config configs/config.yaml
```
3. 추론 (best 모델 자동 로드)
```bash
python inference.py --config configs/config.yaml
```
2. 설정 오버라이드 예시
배치 크기, 에폭 수 변경
`python train.py --override training.batch_size=128 training.max_epochs=50`
모델 파라미터 변경
`python train.py --override model.baller2vec.d_model=256 model.baller2vec.n`