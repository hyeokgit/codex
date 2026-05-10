import os
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

try:
    import OpenDartReader  # type: ignore
except Exception:
    OpenDartReader = None

try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None

FX_TICKER = "KRW=X"
MARKET_TICKERS = {"KOSPI": "^KS11", "KOSDAQ": "^KQ11", "S&P 500": "^GSPC", "NASDAQ": "^IXIC", "USD/KRW": FX_TICKER}

ASSET_TEMPLATE = [
    {"name": "현금", "ticker": "CASH", "qty": 8000000, "currency": "KRW", "type": "cash", "tag": "Core"},
    {"name": "예적금", "ticker": "DEPOSIT", "qty": 6000000, "currency": "KRW", "type": "deposit", "tag": "Core"},
    {"name": "삼성전자우", "ticker": "005935.KS", "qty": 25, "currency": "KRW", "type": "kr_stock", "tag": "Satellite"},
    {"name": "LG전자", "ticker": "066570.KS", "qty": 10, "currency": "KRW", "type": "kr_stock", "tag": "정리 대상"},
    {"name": "NVIDIA", "ticker": "NVDA", "qty": 8, "currency": "USD", "type": "us_stock", "tag": "Satellite"},
    {"name": "VOO", "ticker": "VOO", "qty": 20, "currency": "USD", "type": "us_stock", "tag": "Core"},
]


def apply_theme():
    st.markdown(
        """
        <style>
        .stApp {background: #070d1d; color: #e9efff;}
        section[data-testid='stSidebar'] {background: #141d33; border-right: 1px solid #2b3a66;}
        .card {background:#151f38; border:1px solid #2b3a66; border-radius:12px; padding:14px;}
        .label {font-size:0.85rem; color:#a7b7dd;}
        .value {font-size:1.9rem; font-weight:700;}
        .sub {font-size:0.9rem; color:#6be6a6;}
        </style>
        """,
        unsafe_allow_html=True,
    )


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


def latest_close(ticker: str) -> Optional[float]:
    try:
        h = yf.Ticker(ticker).history(period="5d", interval="1d")
        return safe_float(h["Close"].dropna().iloc[-1]) if not h.empty else None
    except Exception:
        return None


def get_fx_rate() -> float:
    return latest_close(FX_TICKER) or 1350.0


def enrich_assets(df: pd.DataFrame) -> pd.DataFrame:
    fx = get_fx_rate()
    rows = []
    for _, row in df.iterrows():
        if row["type"] in ("cash", "deposit"):
            price, value_krw, roe, per, eps = 1.0, row["qty"], None, None, None
        else:
            price = latest_close(row["ticker"]) or 0.0
            info = yf.Ticker(row["ticker"]).get_info()
            roe = safe_float(info.get("returnOnEquity"))
            per = safe_float(info.get("trailingPE"))
            eps = safe_float(info.get("earningsGrowth"))
            value_krw = row["qty"] * price * (fx if row["currency"] == "USD" else 1)
        rows.append({**row, "price": price, "value_krw": value_krw, "roe_pct": roe * 100 if roe else None, "per": per, "eps_growth_pct": eps * 100 if eps else None})
    return pd.DataFrame(rows)


def styled_portfolio(df: pd.DataFrame):
    show = df[["name", "ticker", "tag", "qty", "price", "value_krw", "pnl_pct", "roe_pct", "per", "eps_growth_pct"]].copy()

    def colorize(v):
        if pd.isna(v):
            return ""
        return "color: #ff6b6b" if v < 0 else "color: #5fe0a2"

    return show.style.format({"price": "{:,.2f}", "value_krw": "₩{:,.0f}", "pnl_pct": "{:.2f}%", "roe_pct": "{:.2f}%", "per": "{:.2f}", "eps_growth_pct": "{:.2f}%"}, na_rep="N/A").map(colorize, subset=["pnl_pct"])


def fetch_dart_notices(corp_codes: List[str]) -> pd.DataFrame:
    api_key = get_secret("DART_API_KEY")
    if not api_key or OpenDartReader is None:
        return pd.DataFrame()
    dart, start, end = OpenDartReader(api_key), date.today() - timedelta(days=7), date.today()
    collected = []
    for code in corp_codes:
        try:
            out = dart.list(code, start=start.strftime("%Y%m%d"), end=end.strftime("%Y%m%d"))
            if out is not None and not out.empty:
                collected.append(out[["rcept_dt", "corp_name", "report_nm"]])
        except Exception:
            continue
    return pd.concat(collected).sort_values("rcept_dt", ascending=False) if collected else pd.DataFrame()


def generate_ai_report(ticker: str, context: str) -> str:
    key = get_secret("GEMINI_API_KEY")
    if not key or genai is None:
        return "GEMINI_API_KEY 또는 google-generativeai 미설치 상태입니다."
    genai.configure(api_key=key)
    res = genai.GenerativeModel("gemini-1.5-flash").generate_content(f"종목 {ticker} 투자 리포트를 한국어 markdown으로 작성:\n{context}")
    return res.text


def main():
    st.set_page_config(page_title="Stock Intelligence", layout="wide")
    apply_theme()

    with st.sidebar:
        st.title("📈 Stock Intelligence")
        st.caption("Phase 1 — Dashboard")
        st.markdown("### 주요 지수")
        for n, t in MARKET_TICKERS.items():
            p = latest_close(t)
            st.write(f"**{n}**  ")
            st.write(f"{p:,.2f}" if p else "N/A")
        st.caption(f"마지막 업데이트: {date.today()}")

    df = enrich_assets(pd.DataFrame(ASSET_TEMPLATE))
    df["cost_basis"] = df["value_krw"] * 0.92
    df["pnl_krw"] = df["value_krw"] - df["cost_basis"]
    df["pnl_pct"] = (df["pnl_krw"] / df["cost_basis"]) * 100

    st.title("💼 포트폴리오 관리")
    st.caption("보유 종목 현황 및 실시간 손익 추적")

    total_invest = float(df["cost_basis"].sum())
    total_eval = float(df["value_krw"].sum())
    pnl = total_eval - total_invest
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='card'><div class='label'>총 투자금액</div><div class='value'>₩{total_invest:,.0f}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='card'><div class='label'>총 평가금액</div><div class='value'>₩{total_eval:,.0f}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='card'><div class='label'>총 평가손익</div><div class='value'>₩{pnl:,.0f}</div><div class='sub'>{(pnl/total_invest*100):.2f}%</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='card'><div class='label'>보유 종목 수</div><div class='value'>{len(df[df['type'].str.contains('stock')])}개</div></div>", unsafe_allow_html=True)

    tabs = st.tabs(["📋 보유 종목", "➕ 종목 추가", "📊 비중 분석", "📄 DART 공시", "🤖 AI 리포트", "♻️ 리밸런싱"])
    with tabs[0]:
        tag_filter = st.multiselect("태그 필터", ["Core", "Satellite", "정리 대상"], default=["Core", "Satellite", "정리 대상"])
        st.dataframe(styled_portfolio(df[df["tag"].isin(tag_filter)]), use_container_width=True)
    with tabs[1]:
        st.info("MVP: 현재는 코드 내 ASSET_TEMPLATE 기반입니다. 다음 단계에서 입력 폼/DB 저장을 연결하세요.")
    with tabs[2]:
        target = {"미국 지수": 0.7, "현금": 0.2, "위성": 0.1}
        current = {
            "미국 지수": df[df["ticker"] == "VOO"]["value_krw"].sum() / total_eval,
            "현금": df[df["type"].isin(["cash", "deposit"])]["value_krw"].sum() / total_eval,
            "위성": df[df["ticker"].isin(["NVDA", "005935.KS", "066570.KS"])]["value_krw"].sum() / total_eval,
        }
        fig = go.Figure()
        fig.add_trace(go.Pie(labels=list(current.keys()), values=list(current.values()), hole=0.58, domain={"x": [0, 0.48]}, name="현재"))
        fig.add_trace(go.Pie(labels=list(target.keys()), values=list(target.values()), hole=0.58, domain={"x": [0.52, 1]}, name="목표"))
        fig.update_layout(paper_bgcolor="#070d1d", font_color="#e9efff")
        st.plotly_chart(fig, use_container_width=True)
    with tabs[3]:
        notices = fetch_dart_notices(["005930", "066570"])
        st.dataframe(notices, use_container_width=True) if not notices.empty else st.warning("DART_API_KEY가 없거나 공시가 없습니다.")
    with tabs[4]:
        pick = st.selectbox("종목 선택", df["ticker"].tolist())
        if st.button("보고서 생성"):
            info = yf.Ticker(pick).get_info()
            ctx = f"회사명:{info.get('shortName')}, PER:{info.get('trailingPE')}, ROE:{info.get('returnOnEquity')}, EPS성장:{info.get('earningsGrowth')}"
            st.markdown(generate_ai_report(pick, ctx))
    with tabs[5]:
        monthly = st.number_input("월 가용 투자금", value=500000, step=100000)
        st.write(f"권장: VOO 중심으로 ₩{monthly*0.7:,.0f}, 현금성 ₩{monthly*0.2:,.0f}, 위성 ₩{monthly*0.1:,.0f}")
        annual_limit = st.number_input("연간 ISA 한도", value=20000000, step=1000000)
        ytd = st.number_input("누적 납입액", value=4000000, step=100000)
        ratio = ytd / annual_limit if annual_limit else 0
        st.progress(min(max(ratio, 0.0), 1.0))
        st.caption(f"ISA 소진율 {ratio*100:.1f}% / 잔여한도 ₩{annual_limit-ytd:,.0f}")


if __name__ == "__main__":
    main()
