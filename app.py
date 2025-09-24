import os
import urllib.parse
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client, Client

# =============== App setup ===============
st.set_page_config(page_title="Amazon Ads Dashboard", page_icon="📊", layout="wide")

# Load env
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error("Missing SUPABASE_URL or SUPABASE_ANON_KEY in .env")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

if "session" not in st.session_state:
    st.session_state.session = None

# =============== Helpers ===============
DATE_CANDIDATES = ("date", "Date", "day", "Day", "order_date", "Report Date", "report_date")

def detect_date_col(df: pd.DataFrame) -> str | None:
    for col in DATE_CANDIDATES:
        if col in df.columns:
            try:
                pd.to_datetime(df[col])
                return col
            except Exception:
                pass
    return None

def load_gsheet_by_sheet_name(sheet_id: str, sheet_name: str) -> pd.DataFrame:
    encoded = urllib.parse.quote(sheet_name)  # space -> %20
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={encoded}"
    return pd.read_csv(url, header=None, dtype=str, low_memory=False)  # raw, no header

def clean_headers(raw: pd.DataFrame, header_idx: int) -> pd.DataFrame:
    # Use chosen row as header
    header_values = raw.iloc[header_idx].fillna("").astype(str).str.strip()
    df = raw.iloc[header_idx + 1 :].reset_index(drop=True)
    df.columns = [c if c else f"col_{i}" for i, c in enumerate(header_values)]
    # Normalize column names
    df.columns = (
        df.columns.str.replace(r"\s+", " ", regex=True)
                  .str.strip()
                  .str.replace(r"[^\w\s\-\/%]", "", regex=True)
    )
    # Try numeric conversion
    for c in df.columns:
        # remove thousands commas then try to numeric
        df[c] = df[c].str.replace(",", "", regex=False) if df[c].dtype == "object" else df[c]
        try:
            df[c] = pd.to_numeric(df[c], errors="ignore")
        except Exception:
            pass
    # Try parse common date cols
    for cand in DATE_CANDIDATES:
        if cand in df.columns:
            try:
                df[cand] = pd.to_datetime(df[cand])
            except Exception:
                pass
    return df

# =============== Auth views ===============
def auth_view():
    st.title("Sign in / Sign up")

    tab_login, tab_signup = st.tabs(["Sign in", "Sign up"])

    with tab_login:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Sign in"):
            try:
                res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                st.session_state.session = res.session
                st.success("Signed in successfully.")
                st.rerun()
            except Exception:
                st.error("Sign in failed. Please check your email or password.")

    with tab_signup:
        email2 = st.text_input("Email", key="signup_email")
        pass2 = st.text_input("Password", type="password", key="signup_pass")
        if st.button("Create account"):
            try:
                supabase.auth.sign_up({"email": email2, "password": pass2})
                st.success("Account created. You can now sign in.")
            except Exception:
                st.error("Sign up failed. Please try again.")

# =============== Dashboard ===============
def dashboard_view():
    st.sidebar.success(f"Signed in: {st.session_state.session.user.email}")
    if st.sidebar.button("Sign out"):
        st.session_state.session = None
        st.rerun()

    st.title("📊 Amazon Ads Dashboard")
    st.caption("Load from Google Sheets by sheet name. If columns look 'Unnamed', pick the correct header row below.")

    # --- YOUR GOOGLE SHEET CONFIG ---
    SHEET_ID = "1E_Zw4rGMgWRS3tvLn11xWjqiwLBpi3u_wJ3N-Jdqyjc"
    SHEET_NAME = "overal analysis"  # exact tab name

    # Load raw (no header) so user can choose header row
    try:
        raw = load_gsheet_by_sheet_name(SHEET_ID, SHEET_NAME)
    except Exception as e:
        st.error(f"Could not load Google Sheet: {e}")
        st.stop()

    st.subheader("Preview (first 10 rows, raw)")
    st.dataframe(raw.head(10), use_container_width=True)

    # Header row picker
    with st.expander("🛠 Fix columns (pick header row)", expanded=True):
        non_empty_counts = raw.notna().sum(axis=1)
        guess_idx = int(non_empty_counts.idxmax()) if len(non_empty_counts) else 0
        header_idx = st.number_input(
            "Header row index (0-based)", min_value=0, max_value=max(0, len(raw)-1),
            value=guess_idx, step=1
        )
        apply_fix = st.button("Apply header")

    if apply_fix or "fixed_df" not in st.session_state:
        st.session_state["fixed_df"] = clean_headers(raw, int(header_idx))
        st.session_state["date_col"] = detect_date_col(st.session_state["fixed_df"])

    df = st.session_state["fixed_df"].copy()
    date_col = st.session_state.get("date_col")

    st.subheader("Cleaned preview")
    st.dataframe(df.head(50), use_container_width=True)

    # KPIs
    def sum_if(col):
        return df[col].fillna(0).sum() if col in df.columns and pd.api.types.is_numeric_dtype(df[col]) else None

    k_impr, k_clk, k_spend, k_sales = map(sum_if, ["Impressions","Clicks","Spend","Sales"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Impressions", f"{int(k_impr):,}" if k_impr is not None else "—")
    c2.metric("Clicks", f"{int(k_clk):,}" if k_clk is not None else "—")
    c3.metric("Spend", f"${k_spend:,.2f}" if k_spend is not None else "—")
    c4.metric("Sales", f"${k_sales:,.2f}" if k_sales is not None else "—")

    if k_impr and k_clk:
        st.caption(f"CTR: {(k_clk / k_impr) * 100:.2f}%")
    if k_spend and k_sales and k_spend > 0:
        roas = k_sales / k_spend
        st.caption(f"ROAS: {roas:.2f}x • ACOS: {(1/roas)*100:.2f}%")

    # Charts
    st.subheader("Charts")
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    y_default = [c for c in numeric_cols if c.lower() in ("sales","clicks","impressions","spend")][:2] or numeric_cols[:2]
    y_cols = st.multiselect("Y-axis: numeric columns to plot", options=numeric_cols, default=y_default)

    x_options = []
    if date_col and date_col in df.columns:
        x_options.append(date_col)
    x_options += non_numeric_cols
    x_options.append("(Row index)")
    x_axis = st.selectbox("X-axis", options=x_options, index=0 if date_col else len(x_options)-1)

    plot_df = df.copy()
    if x_axis == "(Row index)":
        plot_df = plot_df.reset_index().rename(columns={"index": "row"}).set_index("row")
    else:
        if x_axis in df.columns:
            # try to make datetime if it looks like date
            try:
                plot_df[x_axis] = pd.to_datetime(plot_df[x_axis])
            except Exception:
                pass
            try:
                plot_df = plot_df.sort_values(x_axis).set_index(x_axis)
            except Exception:
                plot_df = plot_df.reset_index().rename(columns={"index": "row"}).set_index("row")

    if y_cols:
        st.line_chart(plot_df[y_cols])
        st.bar_chart(plot_df[y_cols])
    else:
        st.info("Select at least one numeric column for Y-axis.")

# =============== Router ===============
if st.session_state.session:
    dashboard_view()
else:
    auth_view()
