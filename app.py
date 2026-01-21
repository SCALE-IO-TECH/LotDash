import re
import numpy as np
import pandas as pd
import streamlit as st

# ---- Official CSV sources (National Lottery site) ----
LOTTO_CSV_URL = "https://www.national-lottery.co.uk/results/lotto/draw-history/csv"
EUROMILLIONS_CSV_URL = "https://www.national-lottery.co.uk/results/euromillions/draw-history/csv"

st.set_page_config(page_title="UK Lottery Dashboard (Stats + Picks)", layout="wide")

st.title("UK Lottery Dashboard (Stats + Entertainment Picks)")
st.caption(
    "This dashboard shows historical stats and generates entertainment picks. "
    "It does NOT predict lottery draws or improve your odds."
)

# ---------------- Helpers ----------------
@st.cache_data(ttl=60 * 60)  # cache for 1 hour
def load_csv(url: str) -> pd.DataFrame:
    # National Lottery CSVs are simple CSVs without stable headers
    df = pd.read_csv(url, header=None, dtype=str)
    return df

def find_date_col(df: pd.DataFrame) -> int:
    """Pick the column that looks most like a date."""
    best_col = None
    best_nonnull = -1
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
    """
    Heuristic: after the detected date column, find numeric-looking columns and take the first `how_many`.
    """
    num_cols = []
    for c in df.columns:
        if c <= date_col:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        # numeric enough?
        if s.notna().mean() > 0.6:
            num_cols.append(int(c))
    return num_cols[:how_many]

def clean_lotto(df_raw: pd.DataFrame) -> pd.DataFrame:
    date_col = find_date_col(df_raw)
    nums = extract_number_cols_after(df_raw, date_col, how_many=7)  # 6 balls + bonus
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df_raw[date_col], errors="coerce", dayfirst=True, infer_datetime_format=True)
    for i in range(6):
        out[f"b{i+1}"] = pd.to_numeric(df_raw[nums[i]], errors="coerce")
    out["bonus"] = pd.to_numeric(df_raw[nums[6]], errors="coerce") if len(nums) >= 7 else np.nan
    out = out.dropna(subset=["date", "b1", "b2", "b3", "b4", "b5", "b6"]).sort_values("date")
    out = out[out[[f"b{i}" for i in range(1,7)]].apply(lambda r: r.nunique() == 6, axis=1)]
    return out.reset_index(drop=True)

def clean_euromillions(df_raw: pd.DataFrame) -> pd.DataFrame:
    date_col = find_date_col(df_raw)
    nums = extract_number_cols_after(df_raw, date_col, how_many=7)  # 5 balls + 2 stars
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df_raw[date_col], errors="coerce", dayfirst=True, infer_datetime_format=True)
    for i in range(5):
        out[f"b{i+1}"] = pd.to_numeric(df_raw[nums[i]], errors="coerce")
    out["s1"] = pd.to_numeric(df_raw[nums[5]], errors="coerce") if len(nums) >= 6 else np.nan
    out["s2"] = pd.to_numeric(df_raw[nums[6]], errors="coerce") if len(nums) >= 7 else np.nan
    out = out.dropna(subset=["date", "b1", "b2", "b3", "b4", "b5", "s1", "s2"]).sort_values("date")
    out = out[out[[f"b{i}" for i in range(1,6)]].apply(lambda r: r.nunique() == 5, axis=1)]
    out = out[out[["s1","s2"]].apply(lambda r: r.nunique() == 2, axis=1)]
    return out.reset_index(drop=True)

def freq_table(values: np.ndarray, max_n: int) -> pd.DataFrame:
    counts = np.bincount(values.astype(int), minlength=max_n + 1)[1:]  # drop 0
    df = pd.DataFrame({"number": np.arange(1, max_n + 1), "count": counts})
    df["pct"] = df["count"] / df["count"].sum()
    return df.sort_values("count", ascending=False).reset_index(drop=True)

def weighted_scores(draws: pd.DataFrame, cols: list[str], max_n: int, last_n: int, alpha: float) -> np.ndarray:
    """
    Transparent (but not predictive) scoring:
    score = alpha * freq(last_n) + (1-alpha) * freq(all_time)
    """
    all_vals = draws[cols].to_numpy().ravel().astype(int)
    recent = draws.tail(last_n)[cols].to_numpy().ravel().astype(int)

    all_counts = np.bincount(all_vals, minlength=max_n + 1)[1:]
    rec_counts = np.bincount(recent, minlength=max_n + 1)[1:]

    # avoid zeros: add tiny smoothing so sampling always possible
    all_counts = all_counts + 1e-9
    rec_counts = rec_counts + 1e-9

    scores = alpha * (rec_counts / rec_counts.sum()) + (1 - alpha) * (all_counts / all_counts.sum())
    return scores

def sample_without_replacement(probs: np.ndarray, k: int, rng: np.random.Generator) -> list[int]:
    probs = probs / probs.sum()
    picks = rng.choice(np.arange(1, len(probs) + 1), size=k, replace=False, p=probs)
    return sorted(picks.tolist())

# ---------------- Sidebar controls ----------------
st.sidebar.header("Settings")
last_n = st.sidebar.slider("Recency window (last N draws)", 20, 300, 120, 10)
alpha = st.sidebar.slider("Recency weight (alpha)", 0.0, 1.0, 0.65, 0.05)
lines = st.sidebar.slider("How many pick lines", 1, 10, 5, 1)
seed = st.sidebar.number_input("Random seed (change for different picks)", value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Reminder: every valid combination has the same chance of being drawn. "
    "‘Smart picks’ are for fun and transparency, not prediction."
)

rng = np.random.default_rng(int(seed))

# ---------------- Load + clean ----------------
with st.spinner("Loading official draw history…"):
    lotto_raw = load_csv(LOTTO_CSV_URL)
    euro_raw = load_csv(EUROMILLIONS_CSV_URL)

lotto = clean_lotto(lotto_raw)
euro = clean_euromillions(euro_raw)

# ---------------- Layout ----------------
tab1, tab2 = st.tabs(["Lotto", "EuroMillions"])

# ===== Lotto tab =====
with tab1:
    left, right = st.columns([1, 1])

    latest = lotto.iloc[-1]
    with left:
        st.subheader("Latest draw (from CSV)")
        st.write(
            {
                "date": latest["date"].date().isoformat(),
                "balls": [int(latest[f"b{i}"]) for i in range(1,7)],
                "bonus": int(latest["bonus"]) if pd.notna(latest["bonus"]) else None,
            }
        )
        st.metric("Draws loaded", len(lotto))

        st.subheader("Entertainment picks (6 numbers)")
        probs = weighted_scores(lotto, [f"b{i}" for i in range(1,7)], max_n=59, last_n=last_n, alpha=alpha)
        for i in range(lines):
            pick = sample_without_replacement(probs, 6, rng)
            st.write(f"Line {i+1}: {pick}")

    with right:
        st.subheader("Most frequent numbers (all time)")
        all_vals = lotto[[f"b{i}" for i in range(1,7)]].to_numpy().ravel()
        ft = freq_table(all_vals, max_n=59)
        st.dataframe(ft.head(15), use_container_width=True)
        st.subheader("Frequency chart (top 20)")
        chart_df = ft.head(20).sort_values("number")
        st.bar_chart(chart_df.set_index("number")["count"])

    st.subheader("Recent draws")
    show = lotto.tail(20).copy()
    show["date"] = show["date"].dt.date
    st.dataframe(show.sort_values("date", ascending=False), use_container_width=True)

# ===== EuroMillions tab =====
with tab2:
    left, right = st.columns([1, 1])

    latest = euro.iloc[-1]
    with left:
        st.subheader("Latest draw (from CSV)")
        st.write(
            {
                "date": latest["date"].date().isoformat(),
                "balls": [int(latest[f"b{i}"]) for i in range(1,6)],
                "stars": [int(latest["s1"]), int(latest["s2"])],
            }
        )
        st.metric("Draws loaded", len(euro))

        st.subheader("Entertainment picks (5 + 2 stars)")
        ball_probs = weighted_scores(euro, [f"b{i}" for i in range(1,6)], max_n=50, last_n=last_n, alpha=alpha)
        star_probs = weighted_scores(euro, ["s1", "s2"], max_n=12, last_n=last_n, alpha=alpha)

        for i in range(lines):
            balls = sample_without_replacement(ball_probs, 5, rng)
            stars = sample_without_replacement(star_probs, 2, rng)
            st.write(f"Line {i+1}: Balls {balls} | Stars {stars}")

    with right:
        st.subheader("Most frequent balls (all time)")
        all_balls = euro[[f"b{i}" for i in range(1,6)]].to_numpy().ravel()
        ftb = freq_table(all_balls, max_n=50)
        st.dataframe(ftb.head(15), use_container_width=True)

        st.subheader("Most frequent stars (all time)")
        all_stars = euro[["s1","s2"]].to_numpy().ravel()
        fts = freq_table(all_stars, max_n=12)
        st.dataframe(fts.head(12), use_container_width=True)

        st.subheader("Ball frequency chart (top 20)")
        chart_df = ftb.head(20).sort_values("number")
        st.bar_chart(chart_df.set_index("number")["count"])

    st.subheader("Recent draws")
    show = euro.tail(20).copy()
    show["date"] = show["date"].dt.date
    st.dataframe(show.sort_values("date", ascending=False), use_container_width=True)

st.markdown("---")
st.caption(
    "Note: This app uses public draw-history CSVs and simple heuristics for entertainment. "
    "It does not and cannot reliably predict the next draw."
)
