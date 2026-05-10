import datetime as dt
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

VALUATION_FIELDS = {
    "trailingPE": "Trailing P/E",
    "forwardPE": "Forward P/E",
    "priceToBook": "P/B",
    "priceToSalesTrailing12Months": "P/S",
    "enterpriseToRevenue": "EV/Revenue",
    "enterpriseToEbitda": "EV/EBITDA",
}


def safe_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        v = float(value)
        if np.isfinite(v):
            return v
    except Exception:
        return None
    return None


def get_current_multiples(ticker: str) -> Dict[str, Optional[float]]:
    info = yf.Ticker(ticker).get_info()
    return {k: safe_float(info.get(k)) for k in VALUATION_FIELDS}


def derive_peers(ticker: str, manual_peers: List[str], limit: int = 6) -> List[str]:
    base = ticker.upper().strip()
    peers = [p.upper().strip() for p in manual_peers if p.strip()]
    peers = [p for p in peers if p != base]
    return peers[:limit]


def fetch_peer_average(peer_tickers: List[str]) -> Dict[str, Optional[float]]:
    if not peer_tickers:
        return {k: None for k in VALUATION_FIELDS}

    rows = []
    for p in peer_tickers:
        try:
            rows.append(get_current_multiples(p))
        except Exception:
            continue

    if not rows:
        return {k: None for k in VALUATION_FIELDS}

    df = pd.DataFrame(rows)
    return {c: safe_float(df[c].mean(skipna=True)) for c in df.columns}


def historical_ps_pb_average(ticker: str, years: int = 5) -> Dict[str, Optional[float]]:
    t = yf.Ticker(ticker)
    end = dt.date.today()
    start = dt.date(end.year - years, end.month, end.day)

    price = t.history(start=start, end=end, interval="1mo", auto_adjust=False)["Close"].dropna()
    if price.empty:
        return {"priceToBook": None, "priceToSalesTrailing12Months": None}

    info = t.get_info()
    shares = safe_float(info.get("sharesOutstanding"))
    if shares is None or shares <= 0:
        return {"priceToBook": None, "priceToSalesTrailing12Months": None}

    market_cap = price * shares

    q_fin = t.quarterly_financials
    q_bs = t.quarterly_balance_sheet

    revenue = None
    equity = None
    if not q_fin.empty and "Total Revenue" in q_fin.index:
        rev = q_fin.loc["Total Revenue"].sort_index()
        revenue = rev.rolling(4).sum().dropna()
    if not q_bs.empty:
        for key in ["Stockholders Equity", "Total Equity Gross Minority Interest", "Total Stockholder Equity"]:
            if key in q_bs.index:
                equity = q_bs.loc[key].sort_index().dropna()
                break

    if revenue is not None and not revenue.empty:
        rev_m = revenue.reindex(pd.to_datetime(price.index), method="ffill")
        ps = (market_cap / rev_m).replace([np.inf, -np.inf], np.nan).dropna()
        ps_avg = safe_float(ps.mean())
    else:
        ps_avg = None

    if equity is not None and not equity.empty:
        eq_m = equity.reindex(pd.to_datetime(price.index), method="ffill")
        pb = (market_cap / eq_m).replace([np.inf, -np.inf], np.nan).dropna()
        pb_avg = safe_float(pb.mean())
    else:
        pb_avg = None

    return {"priceToBook": pb_avg, "priceToSalesTrailing12Months": ps_avg}


def build_comparison_df(current, hist, peers):
    rows = []
    for field, label in VALUATION_FIELDS.items():
        rows.append(
            {
                "Metric": label,
                "Current": current.get(field),
                "Historical Avg (5Y)": hist.get(field),
                "Peers Avg": peers.get(field),
            }
        )
    return pd.DataFrame(rows)


def plot_comparison(df: pd.DataFrame, ticker: str):
    fig = go.Figure()
    for col in ["Current", "Historical Avg (5Y)", "Peers Avg"]:
        fig.add_trace(
            go.Bar(
                x=df["Metric"],
                y=df[col],
                name=col,
                text=[f"{v:.2f}" if pd.notna(v) else "N/A" for v in df[col]],
                textposition="outside",
            )
        )

    fig.update_layout(
        barmode="group",
        title=f"{ticker.upper()} 밸류에이션 비교",
        xaxis_title="지표",
        yaxis_title="배수(Multiple)",
        legend_title="비교 기준",
        height=560,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def main():
    st.set_page_config(page_title="미국 주식 밸류에이션 대시보드", layout="wide")
    st.title("📊 미국 주식 밸류에이션 대시보드 (Yahoo Finance)")
    st.caption("티커 입력 후 현재 밸류에이션을 5년 평균과 경쟁사 평균과 비교합니다.")

    col1, col2 = st.columns([2, 3])
    with col1:
        ticker = st.text_input("미국 상장 티커", value="AAPL").strip().upper()
        peer_input = st.text_input(
            "경쟁사 티커(쉼표 구분, 비워두면 직접 입력 필요)",
            value="MSFT,GOOGL,AMZN,META",
        )
        run = st.button("분석 실행", type="primary")

    with col2:
        st.markdown(
            "- 현재값: Yahoo Finance의 최신 지표\n"
            "- 5년 평균: 월별 시가총액/분기 재무데이터 기반 추정(P/B, P/S만 제공)\n"
            "- 경쟁사 평균: 입력한 동종 티커 평균"
        )

    if run and ticker:
        with st.spinner("데이터 불러오는 중..."):
            manual_peers = [x.strip() for x in peer_input.split(",") if x.strip()]
            peers = derive_peers(ticker, manual_peers)

            current = get_current_multiples(ticker)
            hist_extra = historical_ps_pb_average(ticker, years=5)
            hist = {k: None for k in VALUATION_FIELDS}
            hist.update(hist_extra)
            peer_avg = fetch_peer_average(peers)

            df = build_comparison_df(current, hist, peer_avg)

        st.subheader(f"{ticker} 밸류에이션 비교표")
        st.dataframe(df, use_container_width=True)

        fig = plot_comparison(df, ticker)
        st.plotly_chart(fig, use_container_width=True)

        missing_hist = df[df["Historical Avg (5Y)"].isna()]["Metric"].tolist()
        if missing_hist:
            st.info(f"5년 평균 데이터 부재: {', '.join(missing_hist)}")

        st.caption(f"경쟁사 티커: {', '.join(peers) if peers else '없음'}")


if __name__ == "__main__":
    main()
