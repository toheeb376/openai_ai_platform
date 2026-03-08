# ============================================================
# OpenAI API Platform Intelligence Dashboard
# Built with Streamlit + Plotly | Dark Intelligence Theme
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import base64
import os

# ── Page Config (must be first Streamlit call) ──────────────
st.set_page_config(
    page_title="OpenAI API Platform Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global Color Palette ────────────────────────────────────
BG_APP       = "rgb(0,0,0)"
BG_SIDEBAR   = "rgb(10,10,10)"
BG_PANEL     = "rgb(18,18,18)"
BG_CHART     = "rgb(12,12,12)"
BORDER       = "rgb(40,40,40)"
WHITE        = "rgb(255,255,255)"

CHART_COLORS = [
    "#FFFFFF", "#10A37F", "#1A7FE8",
    "#F4A261", "#ACACAC", "#E63946",
    "#A8DADC", "#457B9D", "#F1FAEE",
]

# ── Custom CSS Injection ────────────────────────────────────
st.markdown("""
<style>
  /* ── App background ── */
  .stApp { background-color: rgb(0,0,0) !important; }
  .main .block-container { background-color: rgb(0,0,0) !important; padding: 1rem 2rem; }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background-color: rgb(10,10,10) !important;
    border-right: 1px solid rgb(40,40,40);
  }
  section[data-testid="stSidebar"] * { color: rgb(255,255,255) !important; }
  section[data-testid="stSidebar"] .stSelectbox > div > div,
  section[data-testid="stSidebar"] .stMultiSelect > div > div {
    background-color: rgb(18,18,18) !important;
    border: 1px solid rgb(40,40,40) !important;
    color: rgb(255,255,255) !important;
  }

  /* ── Global text ── */
  * { color: rgb(255,255,255) !important; }
  h1, h2, h3, h4, h5, h6 { color: rgb(255,255,255) !important; }
  p, span, label, div { color: rgb(255,255,255) !important; }

  /* ── KPI metric cards ── */
  div[data-testid="metric-container"] {
    background-color: rgb(18,18,18) !important;
    border: 1px solid rgb(40,40,40) !important;
    border-radius: 8px !important;
    padding: 14px 18px !important;
  }
  div[data-testid="metric-container"] label { color: rgb(172,172,172) !important; font-size: 0.78rem !important; }
  div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: rgb(255,255,255) !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
  }

  /* ── Selectbox / Multiselect / DateInput ── */
  .stSelectbox > div, .stMultiSelect > div {
    background-color: rgb(18,18,18) !important;
    border: 1px solid rgb(40,40,40) !important;
  }
  .stDateInput > div { background-color: rgb(18,18,18) !important; border: 1px solid rgb(40,40,40) !important; }

  /* ── Expander ── */
  details { background-color: rgb(18,18,18) !important; border: 1px solid rgb(40,40,40) !important; border-radius: 6px !important; padding: 6px; }
  summary { color: rgb(255,255,255) !important; font-weight: 600; }

  /* ── Dividers ── */
  hr { border-color: rgb(40,40,40) !important; }

  /* ── Plotly chart container ── */
  .js-plotly-plot .plotly { background-color: rgb(12,12,12) !important; }

  /* ── Dropdown options overlay ── */
  .stSelectbox [data-baseweb="select"] span { color: rgb(255,255,255) !important; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; background: rgb(0,0,0); }
  ::-webkit-scrollbar-thumb { background: rgb(40,40,40); border-radius: 4px; }

  /* ── Hide default Streamlit elements ── */
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helper: standard Plotly dark layout ────────────────────
def dark_layout(title="", height=400, showlegend=True):
    return dict(
        title=dict(text=title, font=dict(color=WHITE, size=14), x=0.01),
        plot_bgcolor=BG_CHART,
        paper_bgcolor=BG_CHART,
        font=dict(color=WHITE, family="Inter, Arial, sans-serif"),
        legend=dict(
            bgcolor=BG_PANEL,
            bordercolor=BORDER,
            borderwidth=1,
            font=dict(color=WHITE),
        ) if showlegend else dict(visible=False),
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(
            gridcolor=BORDER, gridwidth=0.5,
            linecolor=BORDER, tickfont=dict(color=WHITE),
            title_font=dict(color=WHITE),
        ),
        yaxis=dict(
            gridcolor=BORDER, gridwidth=0.5,
            linecolor=BORDER, tickfont=dict(color=WHITE),
            title_font=dict(color=WHITE),
        ),
        hoverlabel=dict(bgcolor="rgb(0,0,0)", font=dict(color=WHITE)),
    )


# ── Load & Cache Data ───────────────────────────────────────
@st.cache_data
def load_data():
    """Load and preprocess the OpenAI API Platform dataset."""
    df = pd.read_excel("openai_ai_platform_dataset.xlsx", engine="openpyxl")

    # ── Standardise column names (strip spaces) ──
    df.columns = [c.strip() for c in df.columns]

    # ── Strip string columns ──
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    # ── Parse dates safely ──
    df["Request Date"] = pd.to_datetime(df["Request Date"], errors="coerce")

    # ── Numeric coercion ──
    num_cols = [
        "Request Count", "Tokens Processed", "Input Tokens", "Output Tokens",
        "API Cost (USD)", "Response Time (ms)", "Success Rate (%)",
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Derived Metrics ──
    df["Token Efficiency Ratio"] = np.where(
        df["Input Tokens"] > 0,
        df["Output Tokens"] / df["Input Tokens"],
        np.nan,
    )
    df["Cost per Request"] = np.where(
        df["Request Count"] > 0,
        df["API Cost (USD)"] / df["Request Count"],
        np.nan,
    )
    df["Cost per 1K Tokens"] = np.where(
        df["Tokens Processed"] > 0,
        (df["API Cost (USD)"] / df["Tokens Processed"]) * 1000,
        np.nan,
    )
    df["Avg Response Time (s)"] = df["Response Time (ms)"] / 1000

    # ── Profitability Tier ──
    def profit_tier(x):
        if pd.isna(x):  return "Unknown"
        if x > 90:      return "High"
        if x >= 70:     return "Medium"
        return "Low"
    df["Profitability Tier"] = df["Success Rate (%)"].apply(profit_tier)

    # ── High-Cost Flag ──
    threshold = df["API Cost (USD)"].quantile(0.90)
    df["High Cost Flag"] = df["API Cost (USD)"] > threshold

    return df


# ── Sidebar ─────────────────────────────────────────────────
def render_sidebar(df):
    """Render sidebar with logo and all interactive filters."""
    with st.sidebar:
        # Logo
        logo_path = "openai_ai_platform.jpg"
        if os.path.exists(logo_path):
            with open(logo_path, "rb") as f:
                logo_b64 = base64.b64encode(f.read()).decode()
            st.markdown(
                f"""
                <div style='text-align:center; padding: 16px 0 8px 0;'>
                  <img src='data:image/jpeg;base64,{logo_b64}'
                       style='width:90px; border-radius:12px;'/>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            "<h2 style='text-align:center; font-size:1rem; color:#ACACAC; "
            "letter-spacing:2px; margin-bottom:6px;'>PLATFORM INTELLIGENCE</h2>",
            unsafe_allow_html=True,
        )
        st.markdown("<hr style='border-color:rgb(40,40,40);margin:4px 0 14px 0;'>",
                    unsafe_allow_html=True)

        # ── Date Range ──
        st.markdown("** Request Date Range**")
        valid_dates = df["Request Date"].dropna()
        min_d = valid_dates.min().date()
        max_d = valid_dates.max().date()
        date_range = st.date_input(
            "Date Range", value=(min_d, max_d),
            min_value=min_d, max_value=max_d,
            label_visibility="collapsed",
        )

        st.markdown("<hr style='border-color:rgb(40,40,40);margin:10px 0;'>",
                    unsafe_allow_html=True)

        def ms(col, label, key):
            opts = sorted([v for v in df[col].unique() if v != "nan"])
            return st.multiselect(label, opts, default=opts, key=key)

        f_payment   = ms("Payment Status",     " Payment Status",     "f_pay")
        f_plan      = ms("Subscription Plan",  " Subscription Plan",  "f_plan")
        f_latency   = ms("Latency Tier",       " Latency Tier",       "f_lat")
        f_region    = ms("Region",             " Region",             "f_reg")
        f_country   = ms("Country",            "️ Country",            "f_cou")
        f_industry  = ms("Industry",           " Industry",           "f_ind")
        f_apptype   = ms("Application Type",   " Application Type",   "f_app")
        f_devplat   = ms("Developer Platform", "️ Developer Platform", "f_dev")
        f_privacy   = ms("Data Privacy Tier",  " Data Privacy Tier",  "f_prv")

    return (
        date_range, f_payment, f_plan, f_latency,
        f_region, f_country, f_industry, f_apptype,
        f_devplat, f_privacy,
    )


# ── Apply Filters ───────────────────────────────────────────
def apply_filters(df, date_range, f_payment, f_plan, f_latency,
                  f_region, f_country, f_industry, f_apptype,
                  f_devplat, f_privacy):
    """Filter the dataframe based on all sidebar selections."""
    dff = df.copy()

    # Date filter
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        dff = dff[dff["Request Date"].between(start, end, inclusive="both")]

    # Categorical filters
    filters = {
        "Payment Status":     f_payment,
        "Subscription Plan":  f_plan,
        "Latency Tier":       f_latency,
        "Region":             f_region,
        "Country":            f_country,
        "Industry":           f_industry,
        "Application Type":   f_apptype,
        "Developer Platform": f_devplat,
        "Data Privacy Tier":  f_privacy,
    }
    for col, selection in filters.items():
        if selection:
            dff = dff[dff[col].isin(selection)]

    return dff


# ── KPI Cards ───────────────────────────────────────────────
def render_kpis(dff):
    """Render 8 KPI metric cards."""
    total_requests    = int(dff["Request Count"].sum())
    total_tokens      = int(dff["Tokens Processed"].sum())
    total_cost        = dff["API Cost (USD)"].sum()
    avg_success       = dff["Success Rate (%)"].mean()
    avg_response_ms   = dff["Response Time (ms)"].mean()
    unique_customers  = dff["Customer ID"].nunique()
    pending_payments  = int((dff["Payment Status"] == "Pending").sum())
    avg_cost_1k       = dff["Cost per 1K Tokens"].mean()

    cols = st.columns(8)
    metrics = [
        ("Total API Requests",      f"{total_requests:,}",               ""),
        ("Total Tokens Processed",  f"{total_tokens/1e6:.2f}M",          ""),
        ("Total API Cost (USD)",    f"${total_cost:,.2f}",               ""),
        ("Avg Success Rate",        f"{avg_success:.1f}%",               ""),
        ("Avg Response Time",       f"{avg_response_ms:.0f} ms",         ""),
        ("Unique Customers",        f"{unique_customers:,}",             ""),
        ("Pending Payments",        f"{pending_payments:,}",             ""),
        ("Avg Cost / 1K Tokens",    f"${avg_cost_1k:.4f}",              ""),
    ]
    for col, (label, value, delta) in zip(cols, metrics):
        with col:
            st.metric(label=label, value=value, delta=delta or None)


# ── Chart 1: API Requests Over Time ────────────────────────
def chart_requests_over_time(dff):
    daily = (
        dff.dropna(subset=["Request Date"])
        .groupby(dff["Request Date"].dt.date)["Request Count"]
        .sum()
        .reset_index()
    )
    daily.columns = ["Date", "Request Count"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["Date"], y=daily["Request Count"],
        mode="lines+markers",
        line=dict(color="#10A37F", width=2),
        marker=dict(size=4, color="#10A37F"),
        fill="tozeroy",
        fillcolor="rgba(16,163,127,0.08)",
        name="Requests",
    ))
    layout = dark_layout("API Requests Over Time", height=350, showlegend=False)
    layout["xaxis"]["title"] = "Date"
    layout["yaxis"]["title"] = "Request Count"
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)


# ── Chart 2: Revenue by Region ─────────────────────────────
def chart_revenue_by_region(dff):
    grp = dff.groupby(["Region", "Subscription Plan"])["API Cost (USD)"].sum().reset_index()

    fig = px.bar(
        grp, x="Region", y="API Cost (USD)", color="Subscription Plan",
        barmode="group", color_discrete_sequence=CHART_COLORS,
    )
    layout = dark_layout("Revenue (API Cost) by Region", height=350)
    layout["xaxis"]["title"] = "Region"
    layout["yaxis"]["title"] = "API Cost (USD)"
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)


# ── Chart 3: Top 10 Products by Tokens ─────────────────────
def chart_top_products_tokens(dff):
    grp = (
        dff.groupby("Product")["Tokens Processed"]
        .sum()
        .nlargest(10)
        .reset_index()
        .sort_values("Tokens Processed")
    )

    fig = go.Figure(go.Bar(
        x=grp["Tokens Processed"], y=grp["Product"],
        orientation="h",
        marker=dict(
            color=grp["Tokens Processed"],
            colorscale=[[0, "#1A7FE8"], [1, "#10A37F"]],
            showscale=False,
        ),
    ))
    layout = dark_layout("Top 10 Products by Tokens Processed", height=350, showlegend=False)
    layout["xaxis"]["title"] = "Tokens Processed"
    layout["yaxis"]["title"] = ""
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)


# ── Chart 4: Payment Status Donut ──────────────────────────
def chart_payment_donut(dff):
    grp = dff["Payment Status"].value_counts().reset_index()
    grp.columns = ["Status", "Count"]

    fig = go.Figure(go.Pie(
        labels=grp["Status"], values=grp["Count"],
        hole=0.55,
        marker=dict(colors=["#10A37F", "#1A7FE8", "#F4A261"],
                    line=dict(color="rgb(0,0,0)", width=2)),
        textfont=dict(color=WHITE),
    ))
    layout = dark_layout("Payment Status Breakdown", height=350)
    layout["showlegend"] = True
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)


# ── Chart 5: Success Rate by Industry ──────────────────────
def chart_success_by_industry(dff):
    grp = dff.groupby("Industry")["Success Rate (%)"].mean().reset_index().sort_values("Success Rate (%)", ascending=False)

    fig = go.Figure(go.Bar(
        x=grp["Industry"], y=grp["Success Rate (%)"],
        marker=dict(color=CHART_COLORS[:len(grp)]),
    ))
    layout = dark_layout("Avg Success Rate by Industry", height=350, showlegend=False)
    layout["yaxis"]["title"] = "Success Rate (%)"
    layout["yaxis"]["range"] = [0, 100]
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)


# ── Chart 6: Request Count by Subscription Plan ────────────
def chart_requests_by_plan(dff):
    grp = dff.groupby("Subscription Plan")["Request Count"].sum().reset_index().sort_values("Request Count", ascending=False)

    fig = go.Figure(go.Bar(
        x=grp["Subscription Plan"], y=grp["Request Count"],
        marker=dict(
            color=grp["Request Count"],
            colorscale=[[0, "#1A7FE8"], [1, "#FFFFFF"]],
            showscale=False,
        ),
    ))
    layout = dark_layout("Request Count by Subscription Plan", height=350, showlegend=False)
    layout["yaxis"]["title"] = "Request Count"
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)


# ── Chart 7: Latency Tier Distribution ─────────────────────
def chart_latency_distribution(dff):
    grp = dff["Latency Tier"].value_counts().reset_index()
    grp.columns = ["Tier", "Count"]

    fig = go.Figure(go.Bar(
        x=grp["Tier"], y=grp["Count"],
        marker=dict(color=["#10A37F", "#1A7FE8", "#F4A261"]),
    ))
    layout = dark_layout("Latency Tier Distribution", height=350, showlegend=False)
    layout["yaxis"]["title"] = "Count"
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)


# ── Chart 8: API Cost by Application Type (Treemap) ────────
def chart_cost_by_app_type(dff):
    grp = dff.groupby(["Application Type", "Industry"])["API Cost (USD)"].sum().reset_index()

    fig = px.treemap(
        grp,
        path=["Application Type", "Industry"],
        values="API Cost (USD)",
        color="API Cost (USD)",
        color_continuous_scale=[[0, "#1A7FE8"], [0.5, "#10A37F"], [1, "#FFFFFF"]],
    )
    layout = dark_layout("API Cost by Application Type", height=380)
    layout["coloraxis_showscale"] = False
    fig.update_traces(textfont=dict(color=WHITE))
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)


# ── Chart 9: Developer Platform Usage ──────────────────────
def chart_developer_platform(dff):
    grp = dff.groupby("Developer Platform")["Request Count"].sum().reset_index().sort_values("Request Count", ascending=False)

    fig = go.Figure(go.Bar(
        x=grp["Developer Platform"], y=grp["Request Count"],
        marker=dict(color=CHART_COLORS[:len(grp)]),
    ))
    layout = dark_layout("Developer Platform Usage", height=350, showlegend=False)
    layout["yaxis"]["title"] = "Total Request Count"
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)


# ── Chart 10: 3D Intelligence Scatter ──────────────────────
def chart_3d_scatter(dff):
    plot_df = dff.dropna(subset=["Request Date", "Tokens Processed", "API Cost (USD)"]).copy()
    plot_df["Date Ordinal"] = plot_df["Request Date"].apply(
        lambda x: x.toordinal() if pd.notna(x) else np.nan
    )

    latency_vals = plot_df["Latency Tier"].unique().tolist()
    color_map = {
        "Low Latency": "#10A37F",
        "Priority":    "#1A7FE8",
        "Standard":    "#F4A261",
    }

    fig = go.Figure()
    for tier in latency_vals:
        sub = plot_df[plot_df["Latency Tier"] == tier]
        fig.add_trace(go.Scatter3d(
            x=sub["Date Ordinal"],
            y=sub["Tokens Processed"],
            z=sub["API Cost (USD)"],
            mode="markers",
            name=tier,
            marker=dict(
                size=4,
                color=color_map.get(tier, "#ACACAC"),
                opacity=0.82,
                line=dict(width=0),
            ),
            customdata=sub[["Customer Name", "Product", "Application Type", "Success Rate (%)"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Product: %{customdata[1]}<br>"
                "App Type: %{customdata[2]}<br>"
                "Success Rate: %{customdata[3]:.1f}%<br>"
                "Tokens: %{y:,}<br>"
                "Cost: $%{z:.4f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(
            text="3D Intelligence Scatter — Date × Tokens × Cost",
            font=dict(color=WHITE, size=14), x=0.01,
        ),
        scene=dict(
            bgcolor=BG_CHART,
            xaxis=dict(
                title="Date (Ordinal)", backgroundcolor=BG_CHART,
                gridcolor=BORDER, linecolor=BORDER,
                tickfont=dict(color=WHITE), title_font=dict(color=WHITE),
            ),
            yaxis=dict(
                title="Tokens Processed", backgroundcolor=BG_CHART,
                gridcolor=BORDER, linecolor=BORDER,
                tickfont=dict(color=WHITE), title_font=dict(color=WHITE),
            ),
            zaxis=dict(
                title="API Cost (USD)", backgroundcolor=BG_CHART,
                gridcolor=BORDER, linecolor=BORDER,
                tickfont=dict(color=WHITE), title_font=dict(color=WHITE),
            ),
        ),
        paper_bgcolor=BG_CHART,
        plot_bgcolor=BG_CHART,
        font=dict(color=WHITE),
        legend=dict(
            bgcolor=BG_PANEL, bordercolor=BORDER, borderwidth=1,
            font=dict(color=WHITE),
        ),
        hoverlabel=dict(bgcolor="rgb(0,0,0)", font=dict(color=WHITE)),
        height=520,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Executive Insight Panel ─────────────────────────────────
def render_executive_insights():
    with st.expander("  Executive Intelligence Summary — Click to Expand", expanded=False):
        st.markdown("""
        <div style='background:rgb(18,18,18); border:1px solid rgb(40,40,40);
                    border-radius:8px; padding:20px; line-height:1.8;'>

        <h4 style='color:#10A37F; margin-top:0;'> Platform Performance</h4>
        <p><b>API Response Time & Latency Tier</b> — The distribution of Low / Priority / Standard latency tiers
        directly signals infrastructure health. A growing share of <em>Priority</em> tier usage may indicate
        premium segment growth; elevated Standard latency warrants capacity review by engineering.</p>

        <h4 style='color:#1A7FE8;'> Cost Optimisation</h4>
        <p><b>Cost per 1K Tokens</b> — Variance across products and subscription plans exposes pricing
        inefficiencies. Enterprise plans typically yield lower unit costs at scale. Finance teams should
        benchmark high-cost model endpoints against token throughput targets monthly.</p>

        <h4 style='color:#F4A261;'> Revenue Exposure</h4>
        <p><b>Payment Status Distribution</b> — A high Trial-to-Paid gap signals conversion risk.
        Pending payments exceeding 10% of total records warrant immediate follow-up by the accounts team
        to protect recognised revenue.</p>

        <h4 style='color:#FFFFFF;'> Success Rate by Industry</h4>
        <p><b>Success Rate Patterns</b> — Industries with sub-85% success rates (e.g., Healthcare, Retail)
        may indicate integration complexity or data formatting mismatches. Technical support and onboarding
        teams should prioritise targeted enablement sessions for these verticals.</p>

        <h4 style='color:#ACACAC;'> Prompt Efficiency</h4>
        <p><b>Token Efficiency Ratio</b> (Output ÷ Input tokens) — Ratios below 0.5 indicate
        over-specified prompts or low-value completions. Product teams can use this signal to
        recommend prompt compression workshops, reducing spend without sacrificing output quality.</p>

        <h4 style='color:#10A37F;'> Daily Action Framework</h4>
        <ul>
          <li><b>Engineering</b> — Monitor real-time latency tier shifts; escalate any High Latency surge within 2 hours.</li>
          <li><b>Finance</b> — Review high-cost flag records (top 10th percentile) weekly for anomaly resolution.</li>
          <li><b>Product</b> — Track token efficiency ratio by model to prioritise prompt tuning roadmaps.</li>
          <li><b>Sales</b> — Target Trial accounts with 30-day tenure for conversion outreach.</li>
          <li><b>Support</b> — Focus onboarding resources on industries with lowest success rates.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)


# ── Section Header Helper ───────────────────────────────────
def section_header(text, color="#10A37F"):
    st.markdown(
        f"<h3 style='color:{color}; border-left:3px solid {color}; "
        f"padding-left:10px; margin:20px 0 8px 0; font-size:1rem; "
        f"letter-spacing:0.5px;'>{text}</h3>",
        unsafe_allow_html=True,
    )


# ── Main App ────────────────────────────────────────────────
def main():
    # ── Load data ──
    df = load_data()

    # ── Sidebar filters ──
    (
        date_range, f_payment, f_plan, f_latency,
        f_region, f_country, f_industry, f_apptype,
        f_devplat, f_privacy,
    ) = render_sidebar(df)

    # ── Filter dataframe ──
    dff = apply_filters(
        df, date_range, f_payment, f_plan, f_latency,
        f_region, f_country, f_industry, f_apptype,
        f_devplat, f_privacy,
    )

    # ── Header ──
    st.markdown(
        "<h1 style='font-size:1.5rem; letter-spacing:1px; margin:0 0 4px 0;'>"
        " OpenAI API Platform Intelligence Dashboard</h1>"
        "<p style='color:rgb(172,172,172); font-size:0.82rem; margin:0 0 18px 0;'>"
        "Real-Time Operational Analytics · Powered by Streamlit + Plotly</p>",
        unsafe_allow_html=True,
    )

    # ── Record count badge ──
    st.markdown(
        f"<span style='background:rgb(18,18,18); border:1px solid rgb(40,40,40); "
        f"border-radius:20px; padding:4px 14px; font-size:0.78rem; color:#10A37F;'>"
        f" {len(dff):,} records · {dff['Customer ID'].nunique():,} customers"
        f"</span>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr style='border-color:rgb(40,40,40); margin:12px 0;'>",
                unsafe_allow_html=True)

    # ── KPIs ──
    render_kpis(dff)

    st.markdown("<hr style='border-color:rgb(40,40,40); margin:18px 0 6px 0;'>",
                unsafe_allow_html=True)

    # ── Row 1 ──
    section_header(" Temporal & Regional Analysis", "#10A37F")
    c1, c2 = st.columns(2)
    with c1: chart_requests_over_time(dff)
    with c2: chart_revenue_by_region(dff)

    # ── Row 2 ──
    section_header(" Product & Payment Intelligence", "#1A7FE8")
    c1, c2 = st.columns(2)
    with c1: chart_top_products_tokens(dff)
    with c2: chart_payment_donut(dff)

    # ── Row 3 ──
    section_header(" Industry & Subscription Analysis", "#F4A261")
    c1, c2 = st.columns(2)
    with c1: chart_success_by_industry(dff)
    with c2: chart_requests_by_plan(dff)

    # ── Row 4 ──
    section_header(" Infrastructure & Application Insights", "#ACACAC")
    c1, c2 = st.columns(2)
    with c1: chart_latency_distribution(dff)
    with c2: chart_developer_platform(dff)

    # ── Row 5: Full-width charts ──
    section_header("️ Cost Distribution by Application Type", "#10A37F")
    chart_cost_by_app_type(dff)

    # ── Row 6: 3D Scatter ──
    section_header(" 3D Intelligence Scatter — Date × Tokens × Cost", "#1A7FE8")
    chart_3d_scatter(dff)

    # ── Executive Insights ──
    st.markdown("<hr style='border-color:rgb(40,40,40); margin:20px 0 10px 0;'>",
                unsafe_allow_html=True)
    render_executive_insights()

    # ── Footer ──
    st.markdown(
        "<br><p style='text-align:center; color:rgb(60,60,60); font-size:0.72rem;'>"
        "OpenAI API Platform Intelligence · Built with Streamlit &amp; Plotly · "
        "Dark Intelligence Theme</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()