import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
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

DATA_DIR = Path("data")
PORTFOLIO_CSV = DATA_DIR / "portfolio.csv"
RESEARCH_LOG_CSV = DATA_DIR / "research_log.csv"
RISK_KEYWORDS = ["소송", "담보", "우발", "충당", "불확실", "채무보증", "특수관계"]


def apply_theme():
    st.markdown(
        """
        <style>
        .stApp {background: #060b19; color: #e9efff;}
        section[data-testid='stSidebar'] {background: #111a2f; border-right: 1px solid #2b3a66;}
        .card {background:#141f38; border:1px solid #2b3a66; border-radius:12px; padding:14px;}
        .label {font-size:0.85rem; color:#a7b7dd;}
        .value {font-size:1.7rem; font-weight:700;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_data_files() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    if not PORTFOLIO_CSV.exists():
        pd.DataFrame(
            [
                {"name": "삼성전자", "ticker": "005930.KS", "qty": 10, "avg_price": 70000, "currency": "KRW"},
                {"name": "SK하이닉스", "ticker": "000660.KS", "qty": 5, "avg_price": 180000, "currency": "KRW"},
                {"name": "NVIDIA", "ticker": "NVDA", "qty": 3, "avg_price": 900, "currency": "USD"},
            ]
        ).to_csv(PORTFOLIO_CSV, index=False)
    if not RESEARCH_LOG_CSV.exists():
        pd.DataFrame(columns=["timestamp", "ticker", "summary", "risk_flags", "raw_note"]).to_csv(RESEARCH_LOG_CSV, index=False)


def get_secret(name: str) -> str:
    return st.secrets.get(name, os.getenv(name, ""))


def safe_float(v) -> Optional[float]:
    try:
        f = float(v)
        if np.isfinite(f):
            return f
    except Exception:
        return None
    return None


def get_price(ticker: str) -> Optional[float]:
    try:
        hist = yf.Ticker(ticker).history(period="5d")
        if hist.empty:
            return None
        return safe_float(hist["Close"].dropna().iloc[-1])
    except Exception:
        return None


def fx_rate() -> float:
    return get_price("KRW=X") or 1350.0


def calc_metrics(ticker: str) -> Dict[str, Optional[float]]:
    info = yf.Ticker(ticker).info
    return {
        "pbr": safe_float(info.get("priceToBook")),
        "roe": (safe_float(info.get("returnOnEquity")) or 0.0) * 100,
        "per": safe_float(info.get("trailingPE")),
        "gpa": safe_float(info.get("grossMargins")),
        "shares": safe_float(info.get("sharesOutstanding")),
        "book": safe_float(info.get("bookValue")),
    }


def pbr_band_position(ticker: str) -> str:
    try:
        hist = yf.Ticker(ticker).history(period="5y", interval="1mo")
        info = yf.Ticker(ticker).info
        bvps = safe_float(info.get("bookValue"))
        if hist.empty or not bvps:
            return "N/A"
        pbr_series = hist["Close"] / bvps
        pmin, pmax, pcur = pbr_series.min(), pbr_series.max(), pbr_series.iloc[-1]
        if pcur <= pmin + (pmax - pmin) * 0.2:
            return "Under-valued"
        if pcur >= pmin + (pmax - pmin) * 0.8:
            return "Over-valued"
        return "Fair"
    except Exception:
        return "N/A"


def calc_srim_price(book_value_per_share: Optional[float], roe_pct: float, req_return: float = 0.08) -> Optional[float]:
    if not book_value_per_share:
        return None
    roe = roe_pct / 100
    return book_value_per_share + (book_value_per_share * (roe - req_return) / req_return)


def style_pnl(df: pd.DataFrame):
    def colorize(v):
        if pd.isna(v):
            return ""
        return "color: red" if v >= 0 else "color: dodgerblue"

    return df.style.format({"pnl_pct": "{:.2f}%", "market_value_krw": "₩{:,.0f}", "invested_krw": "₩{:,.0f}"}).map(colorize, subset=["pnl_pct"])


def dart_summary_stub(ticker: str) -> Dict[str, str]:
    if dart is None or not get_secret("DART_API_KEY"):
        return {"business": "DART API 미설정", "finance": "DART API 미설정", "note": "설정 필요"}
    return {
        "business": f"{ticker}의 사업 개요 요약(샘플): 주력 제품/서비스와 시장 점유율 변화.",
        "finance": f"{ticker}의 재무 요약(샘플): 매출/영업이익/ROE 추세 정리.",
        "note": "주석 키워드 점검 필요",
    }


def detect_risk_flags(text: str) -> List[str]:
    return [k for k in RISK_KEYWORDS if k in text]


def gemini_three_line_insight(ticker: str, roe: float, per: Optional[float], pbr: Optional[float]) -> str:
    key = get_secret("GEMINI_API_KEY")
    if not key or genai is None:
        return "Gemini 미설정: ROE 개선 원인을 부채/본업 관점으로 수동 점검하세요."
    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = (
        f"{ticker}의 ROE={roe:.2f}%, PER={per}, PBR={pbr}.")
    prompt += "ROE 상승 원인이 레버리지인지 본업 경쟁력인지 3줄 한국어 요약."
    return model.generate_content(prompt).text


def main():
    st.set_page_config(page_title="Stock Intelligence", layout="wide")
    apply_theme()
    ensure_data_files()

    st.title("📊 Stock Intelligence")
    st.caption("DART + 재무지표 + AI를 결합한 Ubuntu Native 리서치")

    pf = pd.read_csv(PORTFOLIO_CSV)
    fx = fx_rate()
    rows = []
    for _, r in pf.iterrows():
        price = get_price(r["ticker"]) or 0.0
        mult = fx if r["currency"] == "USD" else 1
        invested = r["qty"] * r["avg_price"] * mult
        mv = r["qty"] * price * mult
        pnl_pct = ((mv - invested) / invested * 100) if invested else 0
        rows.append({**r, "price": price, "invested_krw": invested, "market_value_krw": mv, "pnl_pct": pnl_pct})
    df = pd.DataFrame(rows)

    total_invested, total_mv = df["invested_krw"].sum(), df["market_value_krw"].sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("총 투자금", f"₩{total_invested:,.0f}")
    c2.metric("평가 금액", f"₩{total_mv:,.0f}")
    c3.metric("총 수익률", f"{((total_mv-total_invested)/total_invested*100):.2f}%")

    tab1, tab2, tab3 = st.tabs(["종합 대시보드", "가치 진단", "AI 리서치"])

    with tab1:
        st.subheader("포트폴리오 트래커")
        st.dataframe(style_pnl(df), use_container_width=True)
        pie = px.pie(df, names="ticker", values="market_value_krw", title="포트폴리오 비중")
        pie.update_layout(paper_bgcolor="#060b19", font_color="#e9efff")
        st.plotly_chart(pie, use_container_width=True)

    with tab2:
        pick = st.selectbox("종목", df["ticker"].tolist(), key="val_pick")
        m = calc_metrics(pick)
        band = pbr_band_position(pick)
        srim = calc_srim_price(m["book"], m["roe"])
        badge = "🏅 슈퍼 우량" if (m["roe"] > 15 and (m["pbr"] or 9) < 1) else "-"
        st.write(f"PBR: {m['pbr']} | ROE: {m['roe']:.2f}% | PER: {m['per']} | GP/A(대체:GrossMargin): {m['gpa']}")
        st.write(f"5년 PBR 밴드 위치: **{band}**")
        st.write(f"S-RIM 적정주가(주당): **{srim:,.0f}**" if srim else "S-RIM 계산 불가")
        st.write(f"진단 배지: {badge}")

    with tab3:
        pick = st.selectbox("리서치 종목", df["ticker"].tolist(), key="ai_pick")
        ds = dart_summary_stub(pick)
        st.markdown("### 공시 자동 브리핑")
        st.write("사업의 내용:", ds["business"])
        st.write("재무에 관한 사항:", ds["finance"])

        m = calc_metrics(pick)
        ai_note = gemini_three_line_insight(pick, m["roe"], m["per"], m["pbr"])
        flags = detect_risk_flags(f"{ds['business']} {ds['finance']} {ds['note']}")
        st.markdown("### Gemini 3줄 인사이트")
        st.write(ai_note)
        st.markdown("### 리스크 체크")
        st.warning(", ".join(flags) if flags else "탐지된 위험 키워드 없음")

        if st.button("리서치 로그 저장"):
            log = pd.read_csv(RESEARCH_LOG_CSV)
            log.loc[len(log)] = [datetime.now().isoformat(), pick, ai_note[:300], ";".join(flags), ds["note"]]
            log.to_csv(RESEARCH_LOG_CSV, index=False)
            st.success("research_log.csv 저장 완료")

        st.markdown("### 리서치 로그")
        st.dataframe(pd.read_csv(RESEARCH_LOG_CSV).tail(30), use_container_width=True)

    st.sidebar.markdown("### Cron 예시")
    st.sidebar.code("0 16 * * 1-5 cd /workspace/codex && python3 app.py # 장 마감 후 재점검")


if __name__ == "__main__":
    main()
