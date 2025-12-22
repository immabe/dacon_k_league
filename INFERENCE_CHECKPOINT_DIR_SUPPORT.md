## 목표
- `inference.py` 실행 시 `--checkpoint` 인자로 **.ckpt 파일 경로뿐 아니라 디렉토리 경로**도 받을 수 있게 한다.

## 문제 원인
- 현재 `--checkpoint` 값이 디렉토리여도 그대로 `KLeagueLightningModule.load_from_checkpoint(checkpoint_path, ...)`에 전달되어 `IsADirectoryError` 발생.

## 할 일(서브태스크)
- `--checkpoint`가 디렉토리인 경우:
  - 해당 디렉토리를 `model_dir`로 사용
  - `best_model_path.txt`가 있으면 그 경로를 우선 사용
  - 없으면 디렉토리 내 `.ckpt`를 찾아 최신 수정 파일을 선택
- `--checkpoint`가 파일인 경우: 기존 동작 유지
- CLI help 문구를 업데이트해서 “파일 또는 디렉토리 가능”을 명확히 안내
- 기존 재현 커맨드로 정상 로딩되는지 확인


