# The Stock Doctor

간결한 Streamlit 대시보드로 주식의 기술/재무/공시 데이터를 한 화면에서 진단합니다.

## 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 필수/선택 설정

- 선택: `DART_API_KEY` (없으면 공시는 안내 문구로 대체)
- 선택: `GEMINI_API_KEY` (없으면 AI 요약은 기본 문구로 대체)

환경변수 또는 `.streamlit/secrets.toml`로 설정하세요.

```toml
DART_API_KEY = "your_key"
GEMINI_API_KEY = "your_key"
```

## 사용 순서

1. 사이드바에서 시장(한국/미국)과 종목 코드 입력
2. `진단 시작` 클릭
3. 상단 핵심 지표 확인: 현재가, AI 점수, ROE, PBR
4. 기술 차트/재무 상세/공시 리스크를 펼쳐 확인

## 예외 처리

- 시세 로드 실패 시 오류 메시지를 표시하고 중단합니다.
- API 키 미설정/라이브러리 미설치 시 기능을 안전하게 축소합니다.
- `@st.cache_data`로 중복 호출을 줄여 응답 속도를 개선합니다.
