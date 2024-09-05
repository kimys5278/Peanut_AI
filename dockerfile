# 베이스 이미지로 Python 3.10 사용 (필요에 따라 Python 버전 수정 가능)
FROM python:3.10-slim

# 컨테이너 내 작업 디렉토리 설정
WORKDIR /app

# 필요한 패키지 설치
COPY requirements.txt .

# 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 소스 복사
COPY . .

# 포트 노출 (FastAPI 기본 포트는 8000)
EXPOSE 8000

# FastAPI 서버 실행 명령어
CMD ["uvicorn", "peanutAI:app", "--host", "0.0.0.0", "--port", "8000"]
