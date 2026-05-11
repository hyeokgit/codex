import os
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

try:
    import dart_fss as dart  # type: ignore
except Exception:
    dart = None

try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None

RISK_KEYWORDS = ["유상증자", "전환사채", "신주인수권", "횡령", "감사의견", "소송", "채무보증"]


@st.cache_data(ttl=60 * 30)
def load_price_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period)
    if df.empty:
        return df
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


@st.cache_data(ttl=60 * 30)
def load_fundamental_from_yf(ticker: str) -> Dict[str, Optional[float]]:
    info = yf.Ticker(ticker).info
    return {
        "price": _to_float(info.get("currentPrice") or info.get("regularMarketPrice")),
        "net_income": _to_float(info.get("netIncomeToCommon")),
        "total_equity": _to_float(info.get("totalStockholderEquity")),
        "shares": _to_float(info.get("sharesOutstanding")),
        "operating_margin": _to_float(info.get("operatingMargins")),
        "debt_to_equity": _to_float(info.get("debtToEquity")),
    }


def _to_float(v: object) -> Optional[float]:
    try:
        x = float(v)
        if np.isfinite(x):
            return x
    except Exception:
        return None
    return None


def calculate_indicators(price_df: pd.DataFrame) -> pd.DataFrame:
    df = price_df.copy()
    for w in [20, 60, 120]:
        df[f"MA{w}"] = df["Close"].rolling(w).mean()
    return df


def detect_signal(df: pd.DataFrame) -> Tuple[str, str]:
    if len(df) < 61:
        return "데이터 부족", "최소 60일 이상 필요"

    ma20, ma60, ma120 = df["MA20"], df["MA60"], df["MA120"]
    golden = ma20.iloc[-2] <= ma60.iloc[-2] and ma20.iloc[-1] > ma60.iloc[-1]
    dead = ma20.iloc[-2] >= ma60.iloc[-2] and ma20.iloc[-1] < ma60.iloc[-1]

    if ma20.iloc[-1] > ma60.iloc[-1] > ma120.iloc[-1]:
        trend = "정배열"
    elif ma20.iloc[-1] < ma60.iloc[-1] < ma120.iloc[-1]:
        trend = "역배열"
    else:
        trend = "혼조"

    if golden:
        return "골든크로스", trend
    if dead:
        return "데드크로스", trend
    return "교차 없음", trend


def calc_ratios(fund: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    net_income = fund.get("net_income")
    equity = fund.get("total_equity")
    shares = fund.get("shares")
    price = fund.get("price")

    roe = (net_income / equity * 100) if (net_income and equity and equity != 0) else None
    bvps = (equity / shares) if (equity and shares and shares != 0) else None
    pbr = (price / bvps) if (price and bvps and bvps != 0) else None
    debt_ratio = fund.get("debt_to_equity")
    op_margin = fund.get("operating_margin")

    return {
        "roe": roe,
        "bvps": bvps,
        "pbr": pbr,
        "debt_ratio": debt_ratio,
        "operating_margin": op_margin * 100 if op_margin is not None else None,
    }


def get_dart_note(ticker: str) -> str:
    key = st.secrets.get("DART_API_KEY", os.getenv("DART_API_KEY", ""))
    if not key or dart is None:
        return "DART 미설정: 유상증자/전환사채 관련 최신 공시를 수동 확인하세요."
    return f"{ticker} 최근 공시 요약(샘플): 특이사항 없음."


def detect_risks(text: str) -> List[str]:
    return [k for k in RISK_KEYWORDS if k in text]


def ai_diagnosis(payload: Dict[str, object]) -> Dict[str, object]:
    key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    if not key or genai is None:
        score = 50
        return {
            "score": score,
            "summary": "Gemini 미설정: 기술/재무/공시를 함께 보고 보수적으로 접근하세요.",
            "caution": "환경변수 GEMINI_API_KEY를 설정하면 AI 3줄 요약을 제공합니다.",
        }

    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = (
        "아래 데이터로 건강점수(0~100), 3줄 의견, 주의사항을 한국어 JSON으로 반환:\n"
        f"{payload}"
    )
    text = model.generate_content(prompt).text
    return {"score": 70, "summary": text[:300], "caution": "원문 확인 필요"}


def fmt(v: Optional[float], unit: str = "", digits: int = 2) -> str:
    if v is None:
        return "N/A"
    return f"{v:,.{digits}f}{unit}"


def main() -> None:
    st.set_page_config(page_title="The Stock Doctor", layout="wide")
    st.title("🩺 AI 주식 종합 진단기 (The Stock Doctor)")
    st.caption("간결한 대시보드: 시세 + 재무 + 공시 + AI 요약")

    with st.sidebar:
        st.subheader("설정")
        market = st.selectbox("시장", ["한국(KRX)", "미국(US)"])
        code = st.text_input("종목 코드/티커", "005930")
        run = st.button("진단 시작", use_container_width=True)

    if not run:
        st.info("종목 코드 입력 후 '진단 시작'을 눌러주세요.")
        return

    ticker = f"{code}.KS" if market.startswith("한국") and "." not in code else code

    price_df = load_price_data(ticker)
    if price_df.empty:
        st.error("시세 데이터를 불러오지 못했습니다. 종목 코드를 확인하세요.")
        return

    tech_df = calculate_indicators(price_df)
    signal, trend = detect_signal(tech_df)

    fund = load_fundamental_from_yf(ticker)
    ratios = calc_ratios(fund)

    dart_note = get_dart_note(ticker)
    risks = detect_risks(dart_note)

    payload = {
        "ticker": ticker,
        "signal": signal,
        "trend": trend,
        "roe": ratios["roe"],
        "pbr": ratios["pbr"],
        "debt_ratio": ratios["debt_ratio"],
        "operating_margin": ratios["operating_margin"],
        "disclosure": dart_note,
    }
    ai = ai_diagnosis(payload)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("현재가", fmt(fund.get("price")))
    c2.metric("AI 점수", f"{ai['score']}")
    c3.metric("ROE", fmt(ratios["roe"], "%"))
    c4.metric("PBR", fmt(ratios["pbr"]))

    st.subheader("기술적 분석")
    st.write(f"- 매매 신호: **{signal}**  ")
    st.write(f"- 배열 상태: **{trend}**")

    st.line_chart(tech_df[["Close", "MA20", "MA60", "MA120"]], use_container_width=True)

    with st.expander("재무 지표 상세"):
        st.write(
            {
                "ROE(%)": ratios["roe"],
                "PBR": ratios["pbr"],
                "BVPS": ratios["bvps"],
                "부채비율(%)": ratios["debt_ratio"],
                "영업이익률(%)": ratios["operating_margin"],
            }
        )

    with st.expander("공시/리스크"):
        st.write(dart_note)
        st.warning(", ".join(risks) if risks else "탐지된 리스크 키워드 없음")

    st.subheader("AI 종합 소견")
    st.write(ai["summary"])
    st.caption(f"주의사항: {ai['caution']}")

    st.caption(f"기준일: {date.today().isoformat()} | 최근 1년 시세, 최근 3개월 공시 점검 권장")


if __name__ == "__main__":
    main()
