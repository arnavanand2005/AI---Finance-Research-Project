"""
Research showcase for the Explainable AI Credit Risk project.

Presents the findings from notebooks 01-06 (EDA, preprocessing, feature
engineering, model training/ablation, evaluation, SHAP explainability)
as a single narrative report. Reads only from data.py — no dependency
on the raw dataset or trained model artifacts.

Run with: streamlit run streamlit_app/app.py
"""

import plotly.graph_objects as go
import requests
import streamlit as st

import data


# FastAPI backend
API_URL = "http://127.0.0.1:8000"


# ---------------------------------------------------------------------------
# Page config & global styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Explainable Credit Risk AI — Research Showcase",
    page_icon="\U0001F4CA",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Credit / banking visual system
INK = "#EEF4FF"
MUTED = "#91A0BC"
PAPER = "#101B33"
PAPER_2 = "#0A1328"
BORDER = "#26385E"

TEAL = "#22D3C5"       # fintech / primary
CYAN = "#38BDF8"       # data / technology
BLUE = "#5B8CFF"       # banking / trust
PURPLE = "#A78BFA"     # AI / explainability
AMBER = "#F5B942"      # caution
RED = "#FF5C68"        # high risk
GREEN = "#34D399"      # low risk

CHART_FONT = dict(family="Inter, sans-serif", color=INK, size=13)


def apply_chart_theme(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=CHART_FONT,
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    return fig


st.markdown(
    f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Lora:ital,wght@0,500;0,600;0,700;1,500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #08111F;
            --panel: #0F1D31;
            --panel2: #12243C;
            --border: #28415F;
            --text: #EAF2FF;
            --muted: #91A4BD;
            --cyan: #28D7D0;
            --blue: #4F8CFF;
            --purple: #9B7CFF;
            --gold: #F4B942;
            --green: #36C98F;
            --red: #F05D68;
        }}

        .stApp {{
            background: var(--bg);
            color: var(--text);
        }}

        .block-container {{
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }}

        section[data-testid="stSidebar"] {{
            background: #0B1728;
            border-right: 1px solid var(--border);
        }}

        h1, h2, h3, .report-title {{
            color: var(--text) !important;
            font-family: Georgia, serif !important;
        }}

        .kicker {{
            color: var(--cyan);
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            font-size: 0.72rem;
            margin-bottom: 0.5rem;
        }}

        .subtitle {{
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.65;
            max-width: 760px;
        }}

        .simple-hero {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-left: 4px solid var(--cyan);
            border-radius: 12px;
            padding: 1.5rem 1.6rem;
            margin: 0.5rem 0 1.3rem 0;
        }}

        .system-strip {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 1.1rem;
        }}

        .system-item {{
            border: 1px solid var(--border);
            background: #10233B;
            border-radius: 7px;
            padding: 0.4rem 0.7rem;
            color: #B8C8DE;
            font-size: 0.78rem;
        }}

        .system-item.cyan {{
            border-color: rgba(40,215,208,0.45);
            color: var(--cyan);
        }}

        .system-item.blue {{
            border-color: rgba(79,140,255,0.45);
            color: #82A9FF;
        }}

        .system-item.purple {{
            border-color: rgba(155,124,255,0.45);
            color: #B59FFF;
        }}

        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.8rem;
            margin: 1rem 0 1.4rem 0;
        }}

        .metric-card {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-top: 3px solid var(--blue);
            border-radius: 10px;
            padding: 1rem 1.1rem;
        }}

        .metric-card:nth-child(1) {{ border-top-color: var(--cyan); }}
        .metric-card:nth-child(2) {{ border-top-color: var(--blue); }}
        .metric-card:nth-child(3) {{ border-top-color: var(--purple); }}
        .metric-card:nth-child(4) {{ border-top-color: var(--gold); }}

        .metric-value {{
            color: var(--text);
            font-family: Georgia, serif;
            font-size: 1.75rem;
            font-weight: 700;
        }}

        .metric-label {{
            color: var(--muted);
            font-size: 0.8rem;
            margin-top: 0.25rem;
        }}

        .pill {{
            display: inline-block;
            border: 1px solid var(--border);
            background: #102038;
            color: #B9C9DF;
            border-radius: 6px;
            padding: 0.28rem 0.65rem;
            margin: 0.15rem 0.2rem 0.15rem 0;
            font-size: 0.76rem;
        }}

        .card {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1rem 1.2rem;
        }}

        .stage-card {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-left: 3px solid var(--cyan);
            border-radius: 9px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.7rem;
        }}

        .stage-card:nth-child(2n) {{
            border-left-color: var(--purple);
        }}

        .stage-num {{
            color: var(--cyan);
            font-weight: 700;
            font-size: 0.75rem;
            letter-spacing: 0.08em;
        }}

        .stage-title {{
            color: var(--text);
            font-weight: 650;
            margin: 0.2rem 0;
        }}

        .stage-desc {{
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.45;
        }}

        .source-card {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 9px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.6rem;
        }}

        .source-name {{
            color: var(--cyan);
            font-weight: 650;
        }}

        .source-desc {{
            color: var(--muted);
            font-size: 0.83rem;
            margin-top: 0.2rem;
        }}

        .finding-list li {{
            color: var(--text);
            margin-bottom: 0.45rem;
            line-height: 1.5;
        }}

        .case-card {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1rem;
        }}

        .case-prob {{
            color: var(--text);
            font-family: Georgia, serif;
            font-size: 2rem;
            font-weight: 700;
        }}

        .risk-badge {{
            display: inline-block;
            border-radius: 5px;
            padding: 0.2rem 0.6rem;
            font-size: 0.74rem;
            font-weight: 700;
        }}

        @media (max-width: 800px) {{
            .metric-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}

        @media (max-width: 520px) {{
            .metric-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Reusable components
# ---------------------------------------------------------------------------

def section_header(kicker: str, title: str, subtitle: str = ""):
    st.markdown(f'<div class="kicker">{kicker}</div>', unsafe_allow_html=True)
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(f'<div class="subtitle">{subtitle}</div>', unsafe_allow_html=True)
    st.write("")


def metric_row(items):
    cards = "".join(
        f"""<div class="metric-card">
                <div class="metric-value">{it['value']}</div>
                <div class="metric-label">{it['label']}</div>
            </div>"""
        for it in items
    )
    st.markdown(f'<div class="metric-grid">{cards}</div>', unsafe_allow_html=True)


def pill_row(items):
    pills = "".join(f'<span class="pill">{it}</span>' for it in items)
    st.markdown(f'<div>{pills}</div>', unsafe_allow_html=True)


def risk_color(level: str) -> str:
    return {"High Risk": RED, "Moderate Risk": AMBER, "Low Risk": GREEN}.get(level, TEAL)


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------

def page_overview():
    st.markdown(
        """
<div class="simple-hero">
    <div class="kicker">Credit Risk / AI Research</div>
    <h1 class="report-title">Explainable AI for Credit Risk Assessment</h1>
    <div class="subtitle">
        A machine-learning system for predicting loan payment difficulty
        using the Home Credit Default Risk dataset. LightGBM provides the
        prediction while SHAP provides transparent explanations.
    </div>
    <div class="system-strip">
        <span class="system-item cyan">DATA → FEATURES</span>
        <span class="system-item blue">FEATURES → LIGHTGBM</span>
        <span class="system-item purple">MODEL → SHAP</span>
        <span class="system-item">EXPLAINABLE CREDIT AI</span>
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    metric_row(data.HEADLINE_STATS)

    pill_row([
        "Python",
        "scikit-learn",
        "LightGBM",
        "SHAP",
        "FastAPI",
        "Streamlit",
    ])

    st.write("")
    st.write("")

    section_header(
        "How we got here",
        "A six-stage research pipeline",
        "Each stage below is a notebook in the project, run in order — every "
        "number in this report traces back to one of them.",
    )

    cols = st.columns(2)

    for i, stage in enumerate(data.PIPELINE_STAGES):
        with cols[i % 2]:
            st.markdown(
                f"""<div class="stage-card">
                    <div class="stage-num">STAGE {stage['stage']}</div>
                    <div class="stage-title">{stage['title']}</div>
                    <div class="stage-desc">{stage['desc']}</div>
                </div>""",
                unsafe_allow_html=True,
            )


def page_dataset():
    section_header(
        "Notebook 01",
        "Dataset & Exploratory Analysis",
        "The Home Credit dataset is large, imbalanced, and only tells part of "
        "the story on its own — six historical sources fill in the rest.",
    )

    d = data.DATASET_OVERVIEW
    metric_row(
        [
            {"value": f"{d['n_applications']:,}", "label": "Loan applications"},
            {"value": d["n_columns"], "label": "Raw columns"},
            {"value": f"{d['target_positive_pct']}%", "label": "Payment difficulty rate"},
            {"value": d["n_missing_features"], "label": "Features with missing values"},
        ]
    )

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("#### Target class imbalance")
        fig = go.Figure(
            go.Pie(
                labels=["No difficulty (0)", "Payment difficulty (1)"],
                values=[d["target_negative_pct"], d["target_positive_pct"]],
                hole=0.62,
                marker=dict(colors=[TEAL, RED]),
                textinfo="label+percent",
                textfont=CHART_FONT,
            )
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(apply_chart_theme(fig, height=320), width='stretch')

    with right:
        st.markdown("#### Strongest correlates with default")
        feats = [c["feature"] for c in data.TARGET_CORRELATIONS][::-1]
        vals = [c["correlation"] for c in data.TARGET_CORRELATIONS][::-1]
        fig = go.Figure(
            go.Bar(
                x=[abs(v) for v in vals],
                y=feats,
                orientation="h",
                marker_color=[CYAN, BLUE, PURPLE, TEAL, AMBER, RED][:len(feats)],
                text=[f"{v:+.3f}" for v in vals],
                textposition="outside",
            )
        )
        fig.update_layout(
            xaxis=dict(title="|Correlation| with TARGET (all negative)", range=[0, max(abs(v) for v in vals) * 1.35]),
        )
        st.plotly_chart(apply_chart_theme(fig, height=320), width='stretch')

    st.write("")
    st.markdown("#### Six historical data sources enrich the raw application")
    cols = st.columns(3)
    for i, src in enumerate(data.HISTORICAL_SOURCES):
        with cols[i % 3]:
            st.markdown(
                f"""<div class="source-card">
                        <div class="source-name">{src['name']}</div>
                        <div class="source-desc">{src['desc']}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

    st.write("")
    st.markdown("#### Key EDA findings")
    st.markdown(
        '<ul class="finding-list">' + "".join(f"<li>{f}</li>" for f in data.EDA_FINDINGS) + "</ul>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page: Methodology
# ---------------------------------------------------------------------------

def page_methodology():
    section_header(
        "Notebooks 02-03",
        "Preprocessing & Feature Engineering",
        "Turning raw, leaky, high-cardinality data into a clean, leakage-safe "
        "422-feature matrix.",
    )

    left, right = st.columns([1, 1.1])
    with left:
        st.markdown("#### Preprocessing pipeline")
        for step in data.PREPROCESSING_STEPS:
            st.markdown(f"- {step}")
        st.markdown("#### Five engineered application features")
        pill_row(data.ENGINEERED_APPLICATION_FEATURES)

    with right:
        st.markdown("#### Where the final 422 features come from")
        groups = data.FEATURE_GROUP_COUNTS
        fig = go.Figure(
            go.Bar(
                x=[g["n_features"] for g in groups][::-1],
                y=[g["group"] for g in groups][::-1],
                orientation="h",
                marker_color=[TEAL, CYAN, BLUE, PURPLE, AMBER, RED][:len(groups)][::-1],
                text=[g["n_features"] for g in groups][::-1],
                textposition="outside",
            )
        )
        fig.update_layout(
            xaxis=dict(title="Feature count", range=[0, max(g["n_features"] for g in groups) * 1.2]),
        )
        st.plotly_chart(apply_chart_theme(fig, height=340), width='stretch')

    st.info(
        "The target remains highly imbalanced after preprocessing — roughly "
        "**11.4 non-default applicants for every default**. This shaped the "
        "class-weighting and threshold decisions in the next stage.",
        icon="ℹ️",
    )


# ---------------------------------------------------------------------------
# Page: Feature Ablation Study
# ---------------------------------------------------------------------------

def page_ablation():
    section_header(
        "Notebook 04",
        "Feature Ablation Study",
        "Does historical credit behavior actually help? Each bar below adds "
        "one more historical source to the application-only baseline.",
    )

    ab = data.ABLATION_STUDY
    fig = go.Figure(
        go.Scatter(
            x=[a["experiment"] for a in ab],
            y=[a["roc_auc"] for a in ab],
            mode="lines+markers+text",
            line=dict(color=TEAL, width=3),
            marker=dict(size=9, color=TEAL),
            text=[f"{a['roc_auc']:.4f}" for a in ab],
            textposition="top center",
            textfont=CHART_FONT,
        )
    )
    fig.update_layout(
        yaxis_title="Validation ROC-AUC",
        yaxis_range=[0.755, 0.79],
    )
    st.plotly_chart(apply_chart_theme(fig, height=400), width='stretch')

    st.markdown(
        "**Previous Applications** provided the largest single incremental gain "
        "(+0.0072 ROC-AUC). **Bureau Balance** alone added little, but was kept "
        "in the final feature set since it can complement other groups when "
        "combined."
    )

    st.write("")
    st.markdown("#### Model configuration comparison (validation ROC-AUC)")
    cfg = data.CONFIG_COMPARISON
    colors = [TEAL] + [MUTED] * (len(cfg) - 2) + [RED]
    fig2 = go.Figure(
        go.Bar(
            x=[c["roc_auc"] for c in cfg][::-1],
            y=[c["experiment"] for c in cfg][::-1],
            orientation="h",
            marker_color=colors[::-1],
            text=[f"{c['roc_auc']:.4f}" for c in cfg][::-1],
            textposition="outside",
        )
    )
    fig2.update_layout(xaxis_title="ROC-AUC", xaxis_range=[0.70, 0.80])
    st.plotly_chart(apply_chart_theme(fig2, height=340), width='stretch')

    st.warning(
        "Class weighting (bottom bar) hurt performance substantially — "
        "**0.7184 vs. 0.7816 ROC-AUC** — so the final model is left unweighted "
        "and relies on threshold tuning instead.",
        icon="⚠️",
    )

    fm = data.FINAL_MODEL_CONFIG
    st.write("")
    st.markdown("#### Selected final model")
    metric_row(
        [
            {"value": fm["algorithm"], "label": "Algorithm"},
            {"value": fm["n_features"], "label": "Features"},
            {"value": f"{fm['val_roc_auc']:.4f}", "label": "Validation ROC-AUC"},
            {"value": fm["best_iteration"], "label": "Best iteration"},
        ]
    )


# ---------------------------------------------------------------------------
# Page: Model Performance
# ---------------------------------------------------------------------------

def page_performance():
    section_header(
        "Notebook 05",
        "Final Model Performance",
        "Evaluated once, on a held-out test set the model never saw during "
        "training or threshold selection.",
    )

    m = data.FINAL_TEST_METRICS
    metric_row(
        [
            {"value": f"{m['roc_auc']:.4f}", "label": "ROC-AUC"},
            {"value": f"{m['pr_auc']:.4f}", "label": "PR-AUC"},
            {"value": f"{m['recall']*100:.1f}%", "label": "Default recall @ 0.15"},
            {"value": f"{m['accuracy']*100:.1f}%", "label": "Accuracy"},
        ]
    )

    left, right = st.columns([1, 1.1])
    with left:
        st.markdown("#### Confusion matrix (threshold = 0.15)")
        cm = data.CONFUSION_MATRIX
        z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=["Predicted: No Default", "Predicted: Default"],
                y=["Actual: No Default", "Actual: Default"],
                colorscale=[[0, PAPER_2], [1, TEAL]],
                text=z,
                texttemplate="%{text:,}",
                textfont=dict(color=INK, size=15),
                showscale=False,
            )
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(apply_chart_theme(fig, height=330), width='stretch')

    with right:
        st.markdown("#### Precision / Recall / F1 by class")
        cr = data.CLASSIFICATION_REPORT
        fig2 = go.Figure()
        for metric_key, color in [("precision", TEAL), ("recall", AMBER), ("f1", MUTED)]:
            fig2.add_trace(
                go.Bar(
                    name=metric_key.capitalize(),
                    x=[c["class"] for c in cr],
                    y=[c[metric_key] for c in cr],
                    marker_color=color,
                )
            )
        fig2.update_layout(barmode="group", yaxis_range=[0, 1], legend=dict(orientation="h", y=1.15))
        st.plotly_chart(apply_chart_theme(fig2, height=330), width='stretch')

    st.markdown(
        f"""Because the dataset is highly imbalanced (~8% default rate), overall
        accuracy is misleading on its own — a model predicting "no default" for
        everyone would score {(1 - 0.0807)*100:.1f}% accuracy while catching zero
        real defaults. The **0.15 classification threshold**, selected on the
        validation set, trades some precision for materially higher recall:
        the model catches **{m['recall']*100:.1f}%** of actual defaults, versus a
        far lower recall at the default 0.50 threshold."""
    )


# ---------------------------------------------------------------------------
# Page: SHAP Explainability
# ---------------------------------------------------------------------------

def page_shap():
    section_header(
        "Notebook 06",
        "SHAP Explainability",
        "SHAP explains *why* the model makes each prediction — globally across "
        "all applicants, and individually for a single applicant.",
    )

    st.markdown("#### Global feature importance (ranked)")
    ranked = data.SHAP_IMPORTANCE_RANK[::-1]
    pseudo_len = list(range(1, len(ranked) + 1))
    fig = go.Figure(
        go.Bar(
            x=pseudo_len,
            y=[r["feature"] for r in ranked],
            orientation="h",
            marker_color=[TEAL, CYAN, BLUE, PURPLE, AMBER] * 5,
        )
    )
    fig.update_layout(
        xaxis_title="Relative rank (mean |SHAP value|, most influential → top)",
        xaxis=dict(showticklabels=False),
    )
    st.plotly_chart(apply_chart_theme(fig, height=420), width='stretch')
    st.caption(
        "Bars represent ranked order of influence, not exact SHAP magnitudes. "
        "Both application-level fields (EXT_SOURCE scores, annuity) and "
        "engineered historical features (late-payment rate, POS behavior) rank "
        "among the top contributors."
    )

    for finding in data.SHAP_FINDINGS:
        st.markdown(f"- {finding}")

    st.write("")
    st.markdown("#### Two individual applicants, explained")
    cols = st.columns(2)
    for col, case in zip(cols, data.CASE_STUDIES):
        color = risk_color(case["risk_level"])
        with col:
            st.markdown(
                f"""<div class="case-card">
                        <div class="kicker" style="color:{color}">{case['label']}</div>
                        <div class="case-prob" style="color:{color}">{case['predicted_probability']*100:.2f}%</div>
                        <div class="metric-label">predicted default probability</div>
                        <div style="margin-top:0.6rem;">
                            <span class="risk-badge" style="background:{color}22;color:{color};border:1px solid {color}55;">
                                {case['risk_level']}
                            </span>
                        </div>
                        <div style="margin-top:0.9rem; color:{MUTED}; font-size:0.88rem; line-height:1.55;">
                            {case['narrative']}
                        </div>
                    </div>""",
                unsafe_allow_html=True,
            )
            st.write("")
            pill_row(case["top_contributors"])


# ---------------------------------------------------------------------------
# Page: Conclusion & Team
# ---------------------------------------------------------------------------

def page_conclusion():
    section_header(
        "Closing",
        "What this project demonstrates",
        "",
    )
    st.markdown(
        """
        Combining application-level information with six sources of historical
        credit behavior improved default prediction from **0.7630 to 0.7816
        validation ROC-AUC** — and the final model generalized to **0.7884
        ROC-AUC** on an untouched test set. Just as importantly, SHAP makes
        every prediction explainable at both the population level and the
        individual applicant level, which is the difference between a credit
        model that merely scores people and one that can justify its decisions.

        The result is a system that is simultaneously **more accurate**
        (historical data measurably helps) and **more transparent** (SHAP shows
        exactly which factors drove any given decision) — the two properties a
        real-world credit-risk tool needs most.
        """
    )

    st.write("")
    st.markdown("#### Team")
    cols = st.columns(len(data.TEAM))
    for col, member in zip(cols, data.TEAM):
        with col:
            st.markdown(
                f"""<div class="card" style="text-align:center;">
                        <div style="font-family:'Lora',serif; font-weight:600; color:{INK};">{member}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

    st.write("")
    st.markdown("#### Tech stack")
    pill_row(["Python", "Pandas", "scikit-learn", "LightGBM", "SHAP", "FastAPI", "Streamlit"])



# ---------------------------------------------------------------------------
# Loan Prediction
# ---------------------------------------------------------------------------

def page_prediction():
    section_header(
        "Live Model",
        "Credit risk prediction",
        "Enter applicant information and send it to the FastAPI backend. "
        "The backend applies the saved 298 → 422 preprocessing pipeline and "
        "returns the LightGBM default probability.",
    )

    st.markdown(
        """
        <div class="simple-hero">
            <div class="kicker">FastAPI / LightGBM</div>
            <div style="font-size:1.05rem; color:#EAF2FF; font-weight:600;">
                Applicant risk assessment
            </div>
            <div class="subtitle" style="margin-top:0.35rem;">
                This interface is connected to the same final model used for
                the research evaluation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Check whether FastAPI is reachable.
    try:
        health_response = requests.get(
            f"{API_URL}/health",
            timeout=3
        )
        api_online = health_response.status_code == 200
    except requests.RequestException:
        api_online = False

    if api_online:
        st.success("FastAPI backend connected")
    else:
        st.error(
            "FastAPI backend is not reachable. Start it with: "
            "uvicorn main:app --reload"
        )
        st.info("Backend URL: http://127.0.0.1:8000")

    st.markdown("### Applicant information")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            income = st.number_input(
                "Annual income",
                min_value=1.0,
                value=180000.0,
                step=10000.0,
                help="AMT_INCOME_TOTAL",
            )

            credit = st.number_input(
                "Credit amount",
                min_value=1.0,
                value=500000.0,
                step=10000.0,
                help="AMT_CREDIT",
            )

            annuity = st.number_input(
                "Loan annuity",
                min_value=1.0,
                value=25000.0,
                step=1000.0,
                help="AMT_ANNUITY",
            )

            goods_price = st.number_input(
                "Goods price",
                min_value=1.0,
                value=450000.0,
                step=10000.0,
                help="AMT_GOODS_PRICE",
            )

            gender = st.selectbox(
                "Gender",
                ["M", "F"],
            )

        with col2:
            days_birth = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=33,
                step=1,
            )

            days_employed = st.number_input(
                "Employment duration (years)",
                min_value=0.0,
                max_value=50.0,
                value=8.0,
                step=1.0,
            )

            family_status = st.selectbox(
                "Family status",
                [
                    "Married",
                    "Single / not married",
                    "Civil marriage",
                    "Separated",
                    "Widow",
                ],
            )

            education = st.selectbox(
                "Education",
                [
                    "Secondary / secondary special",
                    "Higher education",
                    "Incomplete higher",
                    "Lower secondary",
                    "Academic degree",
                ],
            )

        st.markdown("### External credit indicators")

        ext1_col, ext2_col, ext3_col = st.columns(3)

        with ext1_col:
            ext_source_1 = st.number_input(
                "EXT_SOURCE_1",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
            )

        with ext2_col:
            ext_source_2 = st.number_input(
                "EXT_SOURCE_2",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.01,
            )

        with ext3_col:
            ext_source_3 = st.number_input(
                "EXT_SOURCE_3",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
            )

        submitted = st.form_submit_button(
            "Assess Credit Risk",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return

    # The FastAPI schema expects DAYS_BIRTH and DAYS_EMPLOYED as negative
    # day counts, matching the Home Credit dataset convention.
    days_birth_value = int(round(-days_birth * 365.25))
    days_employed_value = int(round(-days_employed * 365.25))

    payload = {
        "AMT_INCOME_TOTAL": float(income),
        "AMT_CREDIT": float(credit),
        "AMT_ANNUITY": float(annuity),
        "AMT_GOODS_PRICE": float(goods_price),
        "EXT_SOURCE_1": float(ext_source_1),
        "EXT_SOURCE_2": float(ext_source_2),
        "EXT_SOURCE_3": float(ext_source_3),
        "DAYS_BIRTH": days_birth_value,
        "DAYS_EMPLOYED": days_employed_value,
        "CODE_GENDER": gender,
        "NAME_FAMILY_STATUS": family_status,
        "NAME_EDUCATION_TYPE": education,
    }

    with st.spinner("Running the LightGBM risk model..."):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30,
            )
        except requests.RequestException as exc:
            st.error(f"Could not connect to FastAPI: {exc}")
            return

    if response.status_code != 200:
        st.error(f"Prediction failed ({response.status_code})")
        try:
            st.json(response.json())
        except ValueError:
            st.code(response.text)
        return

    result = response.json()

    st.markdown("### Prediction result")

    probability = result["default_probability"]
    percentage = result["default_percentage"]
    risk_level = result["risk_level"]
    prediction = result["prediction"]

    if risk_level == "High Risk":
        accent = RED
    elif risk_level == "Moderate Risk":
        accent = AMBER
    else:
        accent = GREEN

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric("Default probability", f"{percentage:.2f}%")

    with result_col2:
        st.metric(
            "Decision",
            "Default risk" if prediction == 1 else "No default risk",
        )

    with result_col3:
        st.markdown(
            f"""
            <div class="card" style="border-top:3px solid {accent};">
                <div style="color:#91A4BD;font-size:0.8rem;">
                    Risk level
                </div>
                <div style="
                    color:{accent};
                    font-family:Georgia,serif;
                    font-size:1.55rem;
                    font-weight:700;
                    margin-top:0.25rem;
                ">
                    {risk_level}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.progress(
        min(max(float(probability), 0.0), 1.0),
        text=f"Model probability: {percentage:.2f}%"
    )

    st.caption(
        f"Decision threshold: {result['threshold']:.2f}. "
        "The threshold was selected using validation data."
    )

    with st.expander("View API request"):
        st.json(payload)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

PAGES = {
    "Overview": page_overview,
    "Loan Prediction": page_prediction,
    "Dataset & EDA": page_dataset,
    "Methodology": page_methodology,
    "Feature Ablation Study": page_ablation,
    "Model Performance": page_performance,
    "SHAP Explainability": page_shap,
    "Conclusion & Team": page_conclusion,
}

with st.sidebar:
    st.markdown(
        '<div style="font-family:\'Lora\',serif; font-size:1.15rem; font-weight:600; '
        'margin-bottom:0.2rem;">Credit Risk AI</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="color:{MUTED}; font-size:0.82rem; margin-bottom:1.2rem;">Research Showcase</div>',
        unsafe_allow_html=True,
    )
    choice = st.radio("Sections", list(PAGES.keys()), label_visibility="collapsed")
    st.write("")
    st.caption("Home Credit Default Risk dataset · LightGBM · SHAP")

PAGES[choice]()
