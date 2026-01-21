import numpy as np
import pandas as pd
import streamlit as st

# =========================
# Config
# =========================
st.set_page_config(
    page_title="UK Lottery Dashboard",
    page_icon="🎲",
    layout="wide",
)

LOTTO_CSV_URL = "https://www.national-lottery.co.uk/results/lotto/draw-history/csv"
EUROMILLIONS_CSV_URL = "https://www.national-lottery.co.uk/results/euromillions/draw-history/csv"

# =========================
# Styling (simple, slick)
# =========================
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
      h1, h2, h3 {letter-spacing: -0.02em;}
      .subtle {color: rgba(255,255,255,0.65); font-size: 0.95rem;}
      .card {
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 18px;
        padding: 16px 16px;
        background: rgba(255,255,255,0.03);
        box-shadow: 0 8px 22px rgba(0,0,0,0.12);
      }
      .pill {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.14);
        background: rgba(255,255,255,0.04);
        margin: 4px 6px 0 0;
        font-weight: 600;
      }
      .pill.gold { border-color: rgba(255, 215, 0, 0.35); }
      .pill.blue { border-color: rgba(0, 150, 255, 0.35); }
      .small { font-size: 0.9rem; color: rgba(255,255,255,0.7); }
      .divider {height: 10px;}
      [data-testid="stMetricValue"] {font-size: 1.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Helpers
# =========================
@st.cache_data(ttl=60 * 60)
def load_csv(url: str) -> pd.DataFrame:
    return pd.read_csv(url, header=None, dtype=str)

def find_date_col(df: pd.DataFrame) -> int:
    best_col, best_nonnull = None, -1
    for c in df.columns:
        parsed = pd.to_datetime(df[c], errors="coerce", dayfirst=True, infer_datetime_format=True)
        nonnull = parsed.notna().sum()
        if nonnull > best_nonnull:
            best_nonnull = nonnull
            best_col = c
    if best_col is None:
        raise ValueError("Could not detect a date column.")
    return int(best_col)

def extract_number_cols_after(df: pd.DataFrame, date_col: int, how_many: int) -> list[int]:
    num_cols = []
    for c in df.columns:
        if c <= date_col:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() > 0.6:
            num_cols.append(int(c))
    return num_cols[:how_many]

def clean_lotto(df_raw: pd.DataFrame) -> pd.DataFrame:
    date_col = find_date_col(df_raw)
    nums = extract_number_cols_after(df_raw, date_col, how_many=7)  # 6 + bonus
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df_raw[date_col], errors="coerce", dayfirst=True, infer_datetime_format=True)
    for i in range(6):
        out[f"b{i+1}"] = pd.to_numeric(df_raw[nums[i]], errors="coerce")
    out["bonus"] = pd.to_numeric(df_raw[nums[6]], errors="coerce") if len(nums) >= 7 else np.nan
    out = out.dropna(subset=["date"] + [f"b{i}" for i in range(1, 7)]).sort_values("date")
    out = out[out[[f"b{i}" for i in range(1, 7)]].apply(lambda r: r.nunique() == 6, axis=1)]
    return out.reset_index(drop=True)

def clean_euro(df_raw: pd.DataFrame) -> pd.DataFrame:
    date_col = find_date_col(df_raw)
    nums = extract_number_cols_after(df_raw, date_col, how_many=7)  # 5 + 2 stars
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df_raw[date_col], errors="coerce", dayfirst=True, infer_datetime_format=True)
    for i in range(5):
        out[f"b{i+1}"] = pd.to_numeric(df_raw[nums[i]], errors="coerce")
    out["s1"] = pd.to_numeric(df_raw[nums[5]], errors="coerce") if len(nums) >= 6 else np.nan
    out["s2"] = pd.to_numeric(df_raw[nums[6]], errors="coerce") if len(nums) >= 7 else np.nan
    out = out.dropna(subset=["date"] + [f"b{i}" for i in range(1, 6)] + ["s1", "s2"]).sort_values("date")
    out = out[out[[f"b{i}" for i in range(1, 6)]].apply(lambda r: r.nunique() == 5, axis=1)]
    out = out[out[["s1", "s2"]].apply(lambda r: r.nunique() == 2, axis=1)]
    return out.reset_index(drop=True)

def score_distribution(draws: pd.DataFrame, cols: list[str], max_n: int, last_n: int, alpha: float) -> np.ndarray:
    """
    Transparent scoring (NOT prediction):
      score = alpha * freq(last_n) + (1-alpha) * freq(all_time)
    + tiny smoothing to avoid zeros.
    """
    all_vals = draws[cols].to_numpy().ravel().astype(int)
    recent_vals = draws.tail(last_n)[cols].to_numpy().ravel().astype(int)

    all_counts = np.bincount(all_vals, minlength=max_n + 1)[1:]
    rec_counts = np.bincount(recent_vals, minlength=max_n + 1)[1:]

    all_counts = all_counts + 1e-9
    rec_counts = rec_counts + 1e-9

    scores = alpha * (rec_counts / rec_counts.sum()) + (1 - alpha) * (all_counts / all_counts.sum())
    scores = scores / scores.sum()
    return scores

def top_k(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    idx = np.argsort(scores)[::-1][:k]
    return [(int(i + 1), float(scores[i])) for i in idx]

def sample_without_replacement(scores: np.ndarray, k: int, rng: np.random.Generator) -> list[int]:
    scores = scores / scores.sum()
    picks = rng.choice(np.arange(1, len(scores) + 1), size=k, replace=False, p=scores)
    return sorted([int(x) for x in picks.tolist()])

def pill_numbers(nums: list[int], kind: str = "gold"):
    pills = "".join([f'<span class="pill {kind}">{n:02d}</span>' for n in nums])
    st.markdown(pills, unsafe_allow_html=True)

def pill_ranked(items: list[tuple[int, float]], kind: str = "gold"):
    # show number + percent
    pills = "".join([f'<span class="pill {kind}">{n:02d} <span class="small">({p*100:.1f}%)</span></span>' for n,p in items])
    st.markdown(pills, unsafe_allow_html=True)

# =========================
# Sidebar controls
# =========================
st.sidebar.header("Controls")
last_n = st.sidebar.slider("Recency window (last N draws)", 20, 300, 120, 10)
alpha = st.sidebar.slider("Recency weight", 0.0, 1.0, 0.65, 0.05)
lines = st.sidebar.slider("Generated lines", 1, 12, 5, 1)
seed = st.sidebar.number_input("Random seed", value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.caption(
    "This app provides stats + a transparent scoring-based pick generator. "
    "It does **not** predict lottery outcomes."
)

rng = np.random.default_rng(int(seed))

# =========================
# Header
# =========================
st.markdown(
    """
    <div class="card">
      <h1 style="margin:0">🎲 UK Lottery Dashboard</h1>
      <div class="subtle">Official draw-history stats • Slick picks generator • Transparent “top-ranked next numbers” scoring</div>
    </div>
    """,
    unsafe_allow_html=True
)
st.write("")

# =========================
# Load data
# =========================
with st.spinner("Loading official draw history…"):
    lotto_raw = load_csv(LOTTO_CSV_URL)
    euro_raw = load_csv(EUROMILLIONS_CSV_URL)

lotto = clean_lotto(lotto_raw)
euro = clean_euro(euro_raw)

# =========================
# Tabs
# =========================
tab1, tab2 = st.tabs(["🇬🇧 Lotto", "⭐ EuroMillions"])

# ============= Lotto UI =============
with tab1:
    latest = lotto.iloc[-1]
    latest_date = latest["date"].date().isoformat()
    latest_balls = [int(latest[f"b{i}"]) for i in range(1, 7)]
    latest_bonus = int(latest["bonus"]) if pd.notna(latest["bonus"]) else None

    scores = score_distribution(lotto, [f"b{i}" for i in range(1, 7)], max_n=59, last_n=last_n, alpha=alpha)
    top6 = top_k(scores, 6)

    c1, c2, c3 = st.columns([1.2, 1.2, 1])

    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Latest draw")
        st.caption(f"Date: {latest_date}")
        pill_numbers(latest_balls, kind="gold")
        if latest_bonus is not None:
            st.write("Bonus:")
            pill_numbers([latest_bonus], kind="blue")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Top-ranked next 6 numbers")
        st.caption("Highest scores under your recency/frequency weighting (not a real prediction).")
        pill_ranked(top6, kind="gold")

        # Also show “single best line” = top6 numbers
        st.write("Suggested ‘Top Line’ (top 6 numbers):")
        pill_numbers(sorted([n for n, _ in top6]), kind="gold")
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Quick stats")
        st.metric("Draws loaded", f"{len(lotto):,}")
        st.metric("Number range", "1–59")
        st.metric("Recency window", f"last {last_n} draws")
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Generate entertainment lines")
    colA, colB = st.columns([1, 2])
    with colA:
        go = st.button("🎲 Generate lines", use_container_width=True)
    with colB:
        st.caption("These are randomly sampled *using your scoring distribution* (still not predictive).")

    if go:
        for i in range(lines):
            pick = sample_without_replacement(scores, 6, rng)
            st.write(f"Line {i+1}")
            pill_numbers(pick, kind="gold")

    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([1, 1])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Score chart (higher = more ‘favoured’ by your weighting)")
        chart_df = pd.DataFrame({"number": np.arange(1, 60), "score": scores})
        st.line_chart(chart_df.set_index("number"))
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Top 15 by score")
        top15 = top_k(scores, 15)
        df_top = pd.DataFrame(top15, columns=["number", "score"])
        df_top["score_%"] = (df_top["score"] * 100).round(2)
        st.dataframe(df_top[["number", "score_%"]], use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Recent draws")
    recent = lotto.tail(24).copy()
    recent["date"] = recent["date"].dt.date
    st.dataframe(recent.sort_values("date", ascending=False), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ============= EuroMillions UI =============
with tab2:
    latest = euro.iloc[-1]
    latest_date = latest["date"].date().isoformat()
    latest_balls = [int(latest[f"b{i}"]) for i in range(1, 6)]
    latest_stars = [int(latest["s1"]), int(latest["s2"])]

    ball_scores = score_distribution(euro, [f"b{i}" for i in range(1, 6)], max_n=50, last_n=last_n, alpha=alpha)
    star_scores = score_distribution(euro, ["s1", "s2"], max_n=12, last_n=last_n, alpha=alpha)

    top5 = top_k(ball_scores, 5)
    top2s = top_k(star_scores, 2)

    c1, c2, c3 = st.columns([1.2, 1.2, 1])

    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Latest draw")
        st.caption(f"Date: {latest_date}")
        st.write("Balls")
        pill_numbers(latest_balls, kind="gold")
        st.write("Stars")
        pill_numbers(latest_stars, kind="blue")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Top-ranked next numbers")
        st.caption("Highest scores under your weighting (not a real prediction).")

        st.write("Top 5 balls")
        pill_ranked(top5, kind="gold")
        st.write("Top 2 stars")
        pill_ranked(top2s, kind="blue")

        st.write("Suggested ‘Top Line’")
        pill_numbers(sorted([n for n, _ in top5]), kind="gold")
        pill_numbers(sorted([n for n, _ in top2s]), kind="blue")
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Quick stats")
        st.metric("Draws loaded", f"{len(euro):,}")
        st.metric("Ball range", "1–50")
        st.metric("Star range", "1–12")
        st.metric("Recency window", f"last {last_n} draws")
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Generate entertainment lines")
    colA, colB = st.columns([1, 2])
    with colA:
        go = st.button("⭐ Generate Euro lines", use_container_width=True)
    with colB:
        st.caption("Random sampling using your scoring distributions (still not predictive).")

    if go:
        for i in range(lines):
            balls = sample_without_replacement(ball_scores, 5, rng)
            stars = sample_without_replacement(star_scores, 2, rng)
            st.write(f"Line {i+1}")
            st.write("Balls")
            pill_numbers(balls, kind="gold")
            st.write("Stars")
            pill_numbers(stars, kind="blue")

    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([1, 1])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Ball score chart")
        chart_df = pd.DataFrame({"number": np.arange(1, 51), "score": ball_scores})
        st.line_chart(chart_df.set_index("number"))
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Star score chart")
        chart_df = pd.DataFrame({"number": np.arange(1, 13), "score": star_scores})
        st.line_chart(chart_df.set_index("number"))
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Recent draws")
    recent = euro.tail(24).copy()
    recent["date"] = recent["date"].dt.date
    st.dataframe(recent.sort_values("date", ascending=False), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")
st.caption(
    "Reminder: lottery draws are random. This dashboard shows stats and a transparent scoring-based generator for entertainment."
)
