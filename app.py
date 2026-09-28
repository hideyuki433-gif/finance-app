import re
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="銘柄まるごとビューア", page_icon="📈", layout="wide")

DATA_DIR = Path(__file__).parent / "data"


def check_password() -> bool:
    """st.secrets に app_password が設定されている場合のみ、簡易パスワードロックを行う。

    未設定（ローカル開発時など）はロックせずそのまま通す。
    """
    try:
        required = st.secrets.get("app_password")
    except Exception:
        required = None
    if not required:
        return True
    if st.session_state.get("authenticated"):
        return True

    st.title("🔒 銘柄まるごとビューア")
    pw = st.text_input("パスワードを入力してください", type="password")
    if pw:
        if pw == required:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("パスワードが違います。")
    return False


if not check_password():
    st.stop()

PERIOD_DAYS = {
    "1ヶ月": 31,
    "3ヶ月": 92,
    "6ヶ月": 183,
    "1年": 365,
    "3年": 1096,
    "5年": 1827,
}
# 200日移動平均（約290暦日）とRSIの平滑化が表示期間の先頭から安定するよう、余分に取得する日数
WARMUP_DAYS = 420

# 証券コード: 4桁数字、または新形式（2・4桁目に英字、例: 130A）
JP_CODE_RE = re.compile(r"^\d[0-9A-Z]\d[0-9A-Z]$")


def normalize_symbol(raw: str) -> tuple[str, bool]:
    """入力文字列を yfinance 用シンボルに正規化する。戻り値は (symbol, is_jp)。"""
    s = raw.strip().upper()
    if not s:
        return s, False
    if s.endswith(".T"):
        return s, True
    if JP_CODE_RE.match(s):
        return f"{s}.T", True
    return s, False


@st.cache_data(show_spinner=False)
def load_jp_master() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "jp_company_master.csv", dtype={"code": str})
    df["key"] = df["name"].map(lambda s: unicodedata.normalize("NFKC", s).lower())
    return df


@st.cache_data(show_spinner=False)
def load_search_index() -> pd.DataFrame:
    """正式名称に通称（ユニクロ→ファーストリテイリング等）を加えた検索用の表。name は常に正式名称。"""
    master = load_jp_master()[["name", "code", "key"]]
    aliases = pd.read_csv(DATA_DIR / "jp_aliases.csv", dtype={"code": str})
    aliases["key"] = aliases["alias"].map(lambda s: unicodedata.normalize("NFKC", s).lower())
    alias_rows = aliases[["code", "key"]].merge(master[["name", "code"]], on="code")
    return pd.concat([master, alias_rows], ignore_index=True)


@st.cache_data(ttl=3600, show_spinner=False)
def search_yahoo(query: str) -> list[dict]:
    """Yahoo Finance の銘柄検索（英語名・ローマ字・ティッカーに強い。日本語の会社名には非対応）。"""
    try:
        return yf.Search(query, max_results=8).quotes or []
    except Exception:
        return []


def _has_ascii_alnum(s: str) -> bool:
    return any(c.isascii() and c.isalnum() for c in s)


def find_candidates(query: str) -> list[dict]:
    """入力文字列から候補銘柄（会社名・コード・ティッカー）のリストを返す。

    4桁の日本株コードや ".T" 付きシンボルはそのまま一意の候補として返す。
    それ以外は、東証全銘柄マスタ＋通称表での部分一致と、Yahoo Finance 検索
    （英語名・ローマ字・海外ティッカー向け）の結果を統合する。
    """
    q = unicodedata.normalize("NFKC", query).strip()
    if not q:
        return []

    master = load_jp_master()

    direct_symbol, direct_is_jp = normalize_symbol(q)
    if direct_is_jp:
        code = direct_symbol.removesuffix(".T")
        hit = master.loc[master["code"] == code, "name"]
        name = hit.iloc[0] if not hit.empty else None
        label = f"{name}（{code}）" if name else direct_symbol
        return [{"label": label, "symbol": direct_symbol, "is_jp": True, "name": name}]

    candidates: list[dict] = []
    seen: set[str] = set()

    ql = q.lower()
    index = load_search_index()
    hits = index[index["key"].str.contains(ql, regex=False)].copy()
    hits["score"] = np.where(hits["key"] == ql, 3, np.where(hits["key"].str.startswith(ql), 2, 1))
    hits["name_len"] = hits["name"].str.len()
    hits = hits.sort_values(["score", "name_len"], ascending=[False, True])
    for name, code in zip(hits["name"], hits["code"]):
        symbol = f"{code}.T"
        if symbol in seen:
            continue
        seen.add(symbol)
        candidates.append({"label": f"{name}（{code}）", "symbol": symbol, "is_jp": True, "name": name})

    if _has_ascii_alnum(q):
        yahoo_candidates = []
        for item in search_yahoo(q):
            if item.get("quoteType") != "EQUITY":
                continue
            symbol = item.get("symbol")
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            disp_name = item.get("longname") or item.get("shortname") or symbol
            exch = item.get("exchDisp", "")
            yahoo_candidates.append(
                {
                    "label": f"{disp_name}（{symbol}・{exch}）" if exch else f"{disp_name}（{symbol}）",
                    "symbol": symbol,
                    "is_jp": symbol.endswith(".T"),
                    "name": disp_name,
                }
            )
        # 同一企業が複数市場に上場している場合、日本のユーザー向けに東証（.T）を優先表示する
        # （優待・四季報リンクも東証コードでないと機能しないため）。
        yahoo_candidates.sort(key=lambda c: 0 if c["is_jp"] else 1)
        candidates.extend(yahoo_candidates)

    return candidates[:10]


@st.cache_data(ttl=300, show_spinner=False)
def load_history(symbol: str, days: int) -> pd.DataFrame:
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    return yf.Ticker(symbol).history(start=start, auto_adjust=False)


@st.cache_data(ttl=1800, show_spinner=False)
def _load_info_cached(symbol: str) -> dict:
    # 失敗時は例外を送出し、st.cache_data に空データをキャッシュさせない
    # （一時的な通信エラーが30分間ずっと反映され続けるのを防ぐ）。
    info = yf.Ticker(symbol).info
    if not info or len(info) < 5:
        raise RuntimeError(f"insufficient info data for {symbol}")
    return info


def load_info(symbol: str) -> dict:
    try:
        return _load_info_cached(symbol)
    except Exception:
        return {}


def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["SMA25"] = out["Close"].rolling(25).mean()
    out["SMA75"] = out["Close"].rolling(75).mean()
    out["SMA200"] = out["Close"].rolling(200).mean()

    # RSI はワイルダー方式（証券会社アプリや TradingView と同じ平滑化）
    delta = out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rsi = 100 - 100 / (1 + avg_gain / avg_loss)
    out["RSI14"] = rsi.where(avg_loss != 0, 100.0)
    return out


def trend_judgement(df: pd.DataFrame) -> tuple[str, str]:
    """直近終値と移動平均線の位置関係から簡易トレンド判定文を返す（投資助言ではなく参考情報）。"""
    if len(df) < 30 or df["SMA25"].isna().iloc[-1]:
        return "判定不可", "データが不足しています"

    last = df.iloc[-1]
    close = last["Close"]
    sma25 = last["SMA25"]
    sma75 = last.get("SMA75", np.nan)

    if pd.notna(sma75):
        if close > sma25 > sma75:
            return "上昇トレンド", "終値が25日線・75日線を上回り、短期線が長期線の上にあります。"
        if close < sma25 < sma75:
            return "下降トレンド", "終値が25日線・75日線を下回り、短期線が長期線の下にあります。"
    if close > sma25:
        return "やや上昇", "終値が25日移動平均線を上回っています。"
    if close < sma25:
        return "やや下降", "終値が25日移動平均線を下回っています。"
    return "横ばい", "終値が25日移動平均線付近で推移しています。"


def rsi_comment(rsi: float) -> str:
    if pd.isna(rsi):
        return "算出不可"
    if rsi >= 70:
        return f"{rsi:.1f}（買われすぎ水準の目安）"
    if rsi <= 30:
        return f"{rsi:.1f}（売られすぎ水準の目安）"
    return f"{rsi:.1f}（中立）"


def fmt_num(value, suffix="", digits=2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:,.{digits}f}{suffix}"
    return str(value)


def fmt_large(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    v = float(value)
    for unit, threshold in (("兆", 1e12), ("億", 1e8), ("百万", 1e6)):
        if abs(v) >= threshold:
            return f"{v / threshold:,.1f}{unit}"
    return f"{v:,.0f}"


def build_external_links(symbol: str, is_jp: bool) -> list[tuple[str, str]]:
    if is_jp:
        code = symbol.replace(".T", "")
        return [
            ("株探（四季報関連情報）", f"https://kabutan.jp/stock/?code={code}"),
            ("みんかぶ（株主優待）", f"https://minkabu.jp/stock/{code}/yutai"),
            ("Yahoo!ファイナンス（優待・四季報タブあり）", f"https://finance.yahoo.co.jp/quote/{symbol}"),
            ("会社四季報オンライン", f"https://shikiho.toyokeizai.net/stocks/{code}"),
        ]
    return [
        ("Yahoo Finance", f"https://finance.yahoo.com/quote/{symbol}"),
        ("StockAnalysis.com", f"https://stockanalysis.com/stocks/{symbol}/"),
        ("Finviz", f"https://finviz.com/quote.ashx?t={symbol}"),
    ]


def build_price_chart(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="株価",
            increasing_line_color="#e05555",
            decreasing_line_color="#3d7fd6",
        )
    )
    for col, color in (("SMA25", "#f0a500"), ("SMA75", "#7a5cf0"), ("SMA200", "#2ca02c")):
        if col in df and df[col].notna().any():
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, mode="lines", line=dict(width=1.3, color=color)))

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=480,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_volume_chart(df: pd.DataFrame) -> go.Figure:
    colors = np.where(df["Close"] >= df["Open"], "#e05555", "#3d7fd6")
    fig = go.Figure(go.Bar(x=df.index, y=df["Volume"], marker_color=colors, name="出来高"))
    fig.update_layout(height=160, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def build_rsi_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(x=df.index, y=df["RSI14"], name="RSI(14)", mode="lines", line=dict(color="#8844cc")))
    fig.add_hline(y=70, line_dash="dot", line_color="gray")
    fig.add_hline(y=30, line_dash="dot", line_color="gray")
    fig.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10), yaxis_range=[0, 100])
    return fig


def show_chart(fig: go.Figure) -> None:
    # スマホでスワイプしただけで拡大縮小されないよう、ドラッグ操作によるズームを無効化する。
    # 拡大したいときは右上のツールバーから意図的に操作できる。タップでの値表示は有効のまま。
    fig.update_layout(dragmode=False)
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"scrollZoom": False, "displaylogo": False, "modeBarButtonsToRemove": ["select2d", "lasso2d"]},
    )


# ---------------- UI ----------------

st.title("📈 銘柄まるごとビューア")
st.caption("日本株・米国株の銘柄名・コード・ティッカーを入力すると、株価チャート・テクニカル指標・優待/四季報リンクを1画面で確認できます。")

# スマホではサイドバーが畳まれて見つけにくいため、検索欄はメインエリア上部に置く
search_col, pick_col, period_col = st.columns([2, 2, 1])
with search_col:
    raw_input = st.text_input(
        "銘柄名・コード・ティッカーで検索",
        value="トヨタ自動車",
        placeholder="例: トヨタ自動車 / Toyota / 7203 / AAPL",
    )

q_norm = unicodedata.normalize("NFKC", raw_input).strip()
candidates = find_candidates(raw_input)
symbol = None
is_jp = False
no_match = False
candidate_name = None

if len(candidates) == 1:
    symbol, is_jp = candidates[0]["symbol"], candidates[0]["is_jp"]
    candidate_name = candidates[0].get("name")
elif len(candidates) > 1:
    labels = [c["label"] for c in candidates]
    with pick_col:
        chosen_label = st.selectbox(f"候補から選択（{len(candidates)}件）", labels, index=0)
    chosen = candidates[labels.index(chosen_label)]
    symbol, is_jp = chosen["symbol"], chosen["is_jp"]
    candidate_name = chosen.get("name")
elif q_norm and re.fullmatch(r"[A-Za-z0-9.\-^=]+", q_norm):
    # ティッカー／コードらしき文字列のみ、そのままシンボルとして試す
    symbol, is_jp = normalize_symbol(q_norm)
elif q_norm:
    no_match = True

with period_col:
    period_label = st.selectbox("表示期間", list(PERIOD_DAYS.keys()), index=3)

if not raw_input.strip():
    st.info("銘柄名・コード・ティッカーを入力してください。")
    st.stop()

if no_match or not symbol:
    st.error(
        f"「{raw_input}」に一致する銘柄が見つかりませんでした。正式名称（例: オリエンタルランド）や"
        "証券コード／ティッカー（例: 4661, AAPL）でお試しください。"
    )
    st.stop()

period_days = PERIOD_DAYS[period_label]

with st.spinner(f"{symbol} のデータを取得中..."):
    try:
        history = load_history(symbol, period_days + WARMUP_DAYS)
    except Exception:
        history = pd.DataFrame()
    info = load_info(symbol)

if history.empty:
    st.error(
        f"「{raw_input}」のデータが取得できませんでした。銘柄コード／ティッカーを確認するか、"
        "データ提供元の混雑の可能性もあるため、時間をおいて再度お試しください。"
    )
    st.stop()

# 指標は余分に取得したデータ全体で計算し、表示は選択期間に絞る
full = calc_indicators(history)
df = full[full.index >= full.index[-1] - timedelta(days=period_days)]
last = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else last
change = last["Close"] - prev["Close"]
change_pct = (change / prev["Close"] * 100) if prev["Close"] else np.nan

name = info.get("longName") or info.get("shortName") or candidate_name or symbol
currency = info.get("currency", "")

# ---- ヘッダー: 銘柄名・現在値 ----
col1, col2, col3, col4 = st.columns([3, 1.4, 1.4, 1.4])
with col1:
    st.subheader(f"{name}　({symbol})")
    sector = info.get("sector")
    industry = info.get("industry")
    if sector or industry:
        st.caption(" / ".join(filter(None, [sector, industry])))
with col2:
    st.metric("現在値", f"{fmt_num(last['Close'])} {currency}", f"{change:+.2f} ({change_pct:+.2f}%)")
with col3:
    st.metric("出来高", fmt_large(last["Volume"]))
with col4:
    week52_high = df["High"].max()
    week52_low = df["Low"].min()
    st.metric("期間高値/安値", f"{fmt_num(week52_high)} / {fmt_num(week52_low)}")

st.markdown("---")

chart_col, side_col = st.columns([3, 1.2])

with chart_col:
    st.markdown("##### 株価チャート（ローソク足・移動平均線）")
    st.caption("日々の始値・高値・安値・終値と、25日／75日／200日移動平均線（過去株価をならした線）を表示します。")
    show_chart(build_price_chart(df, f"{name} 株価チャート（{period_label}）"))

    st.markdown("##### 出来高")
    st.caption("その日に売買が成立した株数です。株価の動きと出来高が一緒に増えているほど、値動きの信頼度が高いと見られます。")
    show_chart(build_volume_chart(df))

    st.markdown("##### RSI（14日・買われすぎ／売られすぎの目安）")
    st.caption("直近14日間の値上がり・値下がりの比率から算出。70以上は買われすぎ、30以下は売られすぎの目安とされます（点線がその境界）。")
    show_chart(build_rsi_chart(df))

with side_col:
    st.markdown("#### トレンド（参考情報）")
    label, reason = trend_judgement(full)
    st.info(f"**{label}**\n\n{reason}")
    st.caption(f"RSI(14): {rsi_comment(last['RSI14'])}")

    st.markdown("#### 主要指標")
    metrics = [
        ("PER（予想）", fmt_num(info.get("forwardPE"))),
        ("PER（実績）", fmt_num(info.get("trailingPE"))),
        ("PBR", fmt_num(info.get("priceToBook"))),
        ("配当利回り", fmt_num(info.get("dividendYield"), "%")),
        ("時価総額", fmt_large(info.get("marketCap"))),
        ("52週高値", fmt_num(info.get("fiftyTwoWeekHigh"))),
        ("52週安値", fmt_num(info.get("fiftyTwoWeekLow"))),
    ]
    for k, v in metrics:
        st.write(f"**{k}**: {v}")

    st.markdown("#### 株主優待・四季報情報")
    st.caption("公式の無料APIがないため、各情報サイトへのリンクを表示します。")
    for label_link, url in build_external_links(symbol, is_jp):
        st.link_button(label_link, url, use_container_width=True)

st.markdown("---")
st.caption(
    "※ 本アプリのトレンド表示は移動平均線やRSIなど過去データに基づく参考情報であり、"
    "投資助言ではありません。投資判断はご自身の責任で行ってください。"
)
st.caption(
    f"データ取得: Yahoo Finance (yfinance) / "
    f"表示時刻: {datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M:%S')}（日本時間）"
)
