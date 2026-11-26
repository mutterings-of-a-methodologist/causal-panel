import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold
import streamlit as st

try:
    from econml.dml import CausalForestDML
except ImportError:  # pragma: no cover
    CausalForestDML = None

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(
    page_title="German Restaurant VAT Cut (19% → 7%) — Simulated Dining Study",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2.3rem;
        padding-bottom: 2.2rem;
        max-width: 1160px;
    }
    section[data-testid="stSidebar"] {
        background-color: #F3F4F6;
    }
    section[data-testid="stSidebar"] .stRadio label p {
        font-size: 1.02rem;
        font-weight: 500;
    }
    .sidebar-intro {
        font-size: 0.88rem;
        color: #374151;
        line-height: 1.55;
        margin-bottom: 1rem;
    }
    .footnote-box {
        font-size: 0.88rem;
        color: #4B5563;
        line-height: 1.65;
        margin-top: 0.75rem;
    }
    .equation-line {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 6px;
        padding: 8px 14px;
        margin: 10px 0;
        display: block;
        color: #111827;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------------
REGION_A = "Region A: Germany (Bavaria / NRW)"
REGION_B = "Region B: Benelux Border Metro"
REGION_C = "Region C: Southern Europe"
REGION_D = "Region D: Nordics"
ALL_REGIONS = [REGION_A, REGION_B, REGION_C, REGION_D]

REGION_COLORS = {
    REGION_A: "#D62728",
    REGION_B: "#FF9F1C",
    REGION_C: "#2CA02C",
    REGION_D: "#1F77B4",
}

START_DATE = pd.Timestamp("2024-01-01")
POLICY_DATE = pd.Timestamp("2026-01-01")
POLICY_DAY = 731          # days from 2024-01-01 to 2026-01-01
N_DAYS = 1035             # data window ends 2026-11-01 (all simulated dates are in the past)
FEATURES = ["Days_Since_Dining", "Dining_Venues_Visited", "Monthly_Dining_EUR", "Deal_Responsive"]

SEG_WEEKLY = "Weekly Regulars (<7d since dining out)"
SEG_CROSS = "Cross-Shopping Casual Diners (7d+ gap, 3+ venues)"
SEG_OCCASIONAL = "Occasional Neighborhood Diners (7d+ gap, 1–2 venues)"
SEGMENTS = [SEG_WEEKLY, SEG_CROSS, SEG_OCCASIONAL]


# -------------------------------------------------------------------------
# SIMULATED DATA
# -------------------------------------------------------------------------
@st.cache_data
def generate_data(seed: int = 42) -> pd.DataFrame:
    """Simulate diner-level records with a known, heterogeneous treatment effect.

    Each row is one simulated diner observation on a random date (a repeated cross-section,
    not a panel). `True_Effect_pp` is the simulation's ground-truth effect of the VAT cut
    for that diner's profile, used only to check the estimators.
    """
    rng = np.random.default_rng(seed)
    n_pts_per_region = 2800

    rows = []
    for reg in ALL_REGIONS:
        t = rng.uniform(0, N_DAYS, size=n_pts_per_region)
        post = (t >= POLICY_DAY).astype(int)
        is_region_a = int(reg == REGION_A)

        days_since_dining = rng.exponential(scale=18.0, size=n_pts_per_region)
        dining_venues_visited = rng.integers(1, 5, size=n_pts_per_region)
        monthly_dining_eur = rng.lognormal(mean=5.4, sigma=0.55, size=n_pts_per_region)  # ~€220 median
        deal_responsive = (rng.uniform(0, 1, size=n_pts_per_region) < 0.45).astype(int)

        # Heterogeneous response to the VAT cut (applies to Region A after Jan 1, 2026 only):
        # strongest drop among occasional / value-conscious diners, weakest among weekly regulars.
        occasional_mask = ((days_since_dining >= 14.0) & (days_since_dining <= 50.0)).astype(float)
        habitual_diner_mask = (days_since_dining < 7.0).astype(float)
        lapsed_diner_mask = (days_since_dining > 60.0).astype(float)

        ind_te = (
            -1.55
            - 1.85 * occasional_mask
            - 0.45 * (dining_venues_visited >= 3).astype(float)
            - 0.40 * deal_responsive
            + 1.15 * habitual_diner_mask
            + 1.25 * lapsed_diner_mask
            + rng.normal(0, 0.25, size=n_pts_per_region)
        )
        ind_te = ind_te - np.mean(ind_te) - 2.80

        if reg == REGION_A:
            y = 18.5 + 0.0005 * t + ind_te * (is_region_a * post) + rng.normal(0, 0.68, size=n_pts_per_region)
        elif reg == REGION_B:
            y = 17.5 + 0.0005 * t + rng.normal(0, 0.68, size=n_pts_per_region)
        elif reg == REGION_C:
            y = 25.5 - 0.0018 * t + rng.normal(0, 0.72, size=n_pts_per_region)
        else:
            y = 12.0 + 0.0020 * t + rng.normal(0, 0.65, size=n_pts_per_region)

        dates = START_DATE + pd.to_timedelta(t, unit="D")
        for i in range(n_pts_per_region):
            rows.append(
                (
                    reg,
                    dates[i],
                    t[i],
                    post[i],
                    is_region_a,
                    y[i],
                    days_since_dining[i],
                    dining_venues_visited[i],
                    monthly_dining_eur[i],
                    deal_responsive[i],
                    ind_te[i],
                )
            )

    return pd.DataFrame(
        rows,
        columns=[
            "Region",
            "Date",
            "Day",
            "Post",
            "Treated_Region_A",
            "Expected_Pullback_Pct",
            "Days_Since_Dining",
            "Dining_Venues_Visited",
            "Monthly_Dining_EUR",
            "Deal_Responsive",
            "True_Effect_pp",
        ],
    )


df_all = generate_data(seed=42)


def segment_labels(days: pd.Series, venues: pd.Series) -> np.ndarray:
    return np.select([days < 7.0, venues >= 3], [SEG_WEEKLY, SEG_CROSS], default=SEG_OCCASIONAL)


@st.cache_data(show_spinner="Fitting honest causal forest (cross-fitted)…")
def fit_causal_forest(max_gap: int, seed: int = 42):
    """Cross-fitted, honest causal forest (EconML CausalForestDML) for Region A vs. Region B.

    Outcomes are baseline-adjusted (each diner's value minus their region's pre-2026 mean) so the
    forest's average effect is comparable to the DiD estimate under parallel trends.
    """
    ab = df_all[df_all["Region"].isin([REGION_A, REGION_B])]
    pre_means = ab[ab["Post"] == 0].groupby("Region")["Expected_Pullback_Pct"].mean()
    d = ab[(ab["Post"] == 1) & (ab["Days_Since_Dining"] <= max_gap)].copy().reset_index(drop=True)
    d["Y_adj"] = d["Expected_Pullback_Pct"] - d["Region"].map(pre_means)
    d["Diner_Segment"] = segment_labels(d["Days_Since_Dining"], d["Dining_Venues_Visited"])
    d["Estimated_Effect_pp"] = np.nan

    X = d[FEATURES]
    T = d["Treated_Region_A"].to_numpy()
    Y = d["Y_adj"].to_numpy()

    fold_rows = []
    kf = KFold(n_splits=2, shuffle=True, random_state=seed)
    for train_idx, test_idx in kf.split(d):
        est = CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=200, min_samples_leaf=20, random_state=seed),
            model_t=RandomForestClassifier(n_estimators=200, min_samples_leaf=20, random_state=seed),
            discrete_treatment=True,
            n_estimators=500,
            min_samples_leaf=10,
            honest=True,
            cv=2,
            random_state=seed,
        )
        est.fit(Y[train_idx], T[train_idx], X=X.iloc[train_idx])
        d.loc[test_idx, "Estimated_Effect_pp"] = est.effect(X.iloc[test_idx])
        for seg in SEGMENTS:
            mask = (d.loc[test_idx, "Diner_Segment"] == seg).to_numpy()
            if mask.sum() >= 20:
                summ = est.effect_inference(X.iloc[test_idx][mask]).population_summary()
                fold_rows.append(
                    (seg, int(mask.sum()), float(np.ravel(summ.mean_point)[0]), float(np.ravel(summ.stderr_mean)[0]))
                )

    folds = pd.DataFrame(fold_rows, columns=["Diner_Segment", "n", "mean", "se"])
    out = []
    for seg, g in folds.groupby("Diner_Segment"):
        w = g["n"] / g["n"].sum()
        out.append(
            {
                "Diner_Segment": seg,
                "Estimated_pp": float((w * g["mean"]).sum()),
                "SE": float(np.sqrt(((w**2) * (g["se"] ** 2)).sum())),
            }
        )
    seg_est = pd.DataFrame(out)

    seg_df = (
        d.groupby("Diner_Segment", as_index=False)
        .agg(
            Diners=("Estimated_Effect_pp", "count"),
            Avg_Days_Since_Dining=("Days_Since_Dining", "mean"),
            Avg_Venues_Visited=("Dining_Venues_Visited", "mean"),
            True_Effect_pp=("True_Effect_pp", "mean"),
        )
        .merge(seg_est, on="Diner_Segment", how="left")
    )
    seg_df["CI_Low"] = seg_df["Estimated_pp"] - 1.96 * seg_df["SE"]
    seg_df["CI_High"] = seg_df["Estimated_pp"] + 1.96 * seg_df["SE"]
    return seg_df.sort_values("Estimated_pp", ascending=True).reset_index(drop=True)


with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-intro">
            This is an independent portfolio project using <strong>simulated</strong> data modeled on the structure of
            aggregated card-transaction records (no real transaction data).
            It evaluates consumer dining pullback across four European markets from January 2024 through October 2026.
            On January 1, 2026, Germany cut VAT on restaurant food (not beverages) from 19% to 7% (Region A only).
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Analyses")
    selected_view = st.radio(
        label="Analyses",
        options=[
            "Data visualizations",
            "Difference in Difference",
            "Causal Forest",
        ],
        label_visibility="collapsed",
    )

# -------------------------------------------------------------------------
# VIEW 1: DATA VISUALIZATIONS
# -------------------------------------------------------------------------
if selected_view == "Data visualizations":
    c1, c2 = st.columns([2.2, 1.3], vertical_alignment="center")
    with c1:
        visible_regions = st.multiselect(
            "Regions shown on chart",
            options=ALL_REGIONS,
            default=ALL_REGIONS,
        )
    with c2:
        show_pre_trends = st.toggle("Overlay pre-Jan 2026 trend lines & p-values", value=False)

    active_regions = visible_regions if visible_regions else [REGION_A]

    fig, ax = plt.subplots(figsize=(10.5, 5.5), dpi=180)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    for reg in active_regions:
        sub = df_all[df_all["Region"] == reg]
        ax.scatter(
            sub["Date"],
            sub["Expected_Pullback_Pct"],
            s=7,
            alpha=0.65,
            color=REGION_COLORS[reg],
            edgecolors="none",
            label=reg,
        )

    trend_notes = []
    if show_pre_trends:
        pre_df = df_all[df_all["Post"] == 0].copy()
        pre_df["Is_RegA"] = (pre_df["Region"] == REGION_A).astype(int)
        for reg in active_regions:
            sub_pre = pre_df[pre_df["Region"] == reg]
            coef = np.polyfit(sub_pre["Day"], sub_pre["Expected_Pullback_Pct"], 1)
            x_pre_days = np.array([0, POLICY_DAY - 1])
            x_pre_dates = START_DATE + pd.to_timedelta(x_pre_days, unit="D")
            ax.plot(
                x_pre_dates,
                coef[0] * x_pre_days + coef[1],
                color=REGION_COLORS[reg],
                linewidth=2.2,
                linestyle="-",
                label=f"{reg.split(':')[0]} Pre-2026 Slope ({coef[0]*365:+.2f} pp/yr)",
            )
            if reg != REGION_A:
                pair_df = pre_df[pre_df["Region"].isin([REGION_A, reg])].copy()
                pair_df["RegA_x_Day"] = pair_df["Is_RegA"] * pair_df["Day"]
                m_pt = smf.ols("Expected_Pullback_Pct ~ Is_RegA + Day + RegA_x_Day", data=pair_df).fit(cov_type="HC1")
                pval = float(m_pt.pvalues["RegA_x_Day"])
                verdict = "parallel pre-trend (fail to reject)" if pval > 0.05 else "divergent pre-trend (reject)"
                trend_notes.append(f"Region A vs. {reg.split(':')[0]}: p = {pval:.4f} [{verdict}]")

    ax.axvline(POLICY_DATE, color="#444444", linestyle="--", linewidth=1.4, label="Restaurant Food VAT Cut: 19% → 7% (Region A Only)")
    ax.set_title("Simulated Expected Dining-Out Pullback Rate at Restaurant Chains by Region", fontsize=12.5, pad=12)
    ax.set_xlabel("Date", fontsize=10.5)
    ax.set_ylabel("Expected Dining-Out Pullback Rate (%)*", fontsize=10.5)
    ax.set_xlim(pd.Timestamp("2023-12-15"), pd.Timestamp("2026-12-01"))
    ax.set_ylim(10.0, 31.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_yticks([10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0])
    ax.tick_params(axis="both", which="major", labelsize=9.5, direction="out", length=4)
    ax.grid(True, linestyle="-", linewidth=0.4, color="#E5E5E5")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
        spine.set_linewidth(0.8)

    ax.legend(
        title="Region",
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#DDDDDD",
        fontsize=8.2,
        title_fontsize=8.8,
        markerscale=2.0,
    )
    plt.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    if show_pre_trends and trend_notes:
        st.info("**Pre-2026 Parallel Trends Tests (Jan 2024 – Dec 2025):** " + " | ".join(trend_notes))

    st.markdown(
        "Each dot in the chart above is one simulated diner's expected probability of cutting back on casual-dining restaurant visits "
        "in favor of grocery or at-home meals across four European regions (Jan 2024 – Oct 2026). "
        "On January 1, 2026, Germany permanently lowered VAT on restaurant food (not beverages) from 19% to 7% (Region A only)."
    )

    st.divider()

    a_df = df_all[df_all["Region"] == REGION_A]
    b_df = df_all[df_all["Region"] == REGION_B]
    a_pre, a_post = (a_df[a_df["Post"] == v]["Expected_Pullback_Pct"].mean() for v in (0, 1))
    b_pre, b_post = (b_df[b_df["Post"] == v]["Expected_Pullback_Pct"].mean() for v in (0, 1))

    st.markdown(
        f"""
        <div class="footnote-box">
            <strong>* How the simulated outcome is defined:</strong><br>
            Each simulated diner has an expected probability of pulling back from casual-dining restaurant spend in a billing cycle:
            <span class="equation-line">Expected Dining-Out Pullback Rate (%) = 100 × P(Pullback | Region, Date, Diner Profile)</span>
            Because this is a simulation, the rate is drawn directly from a known data-generating process rather than estimated from real
            transactions. With real card data, the analogous outcome would be observed restaurant spending or visit counts.
            <strong>What the simulation produces:</strong> the average Region A diner's rate moves from <strong>{a_pre:.1f}%</strong> before
            January 1, 2026 (19% VAT regime) to <strong>{a_post:.1f}%</strong> after the cut, while the Region B average moves from
            <strong>{b_pre:.1f}%</strong> to <strong>{b_post:.1f}%</strong>. These are raw means; the DiD view adjusts for the gap and drift between regions.
            The simulation assumes the tax saving reaches diners (through prices or promotions); real-world evidence suggests pass-through to menu
            prices has been limited, so the effect in practice could be smaller.
            All records are synthetically generated (<code>seed = 42</code>) and contain no proprietary merchant or cardholder data.
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------------------------------------------------
# VIEW 2: DIFFERENCE IN DIFFERENCES
# -------------------------------------------------------------------------
elif selected_view == "Difference in Difference":
    c1, c2 = st.columns([1.8, 1.5], vertical_alignment="center")
    with c1:
        peer_region = st.selectbox(
            "Comparison market against Region A: Germany (Bavaria / NRW)",
            options=[REGION_B, REGION_C, REGION_D],
            index=0,
        )
    with c2:
        zoom_window = st.selectbox(
            "Time window zoom",
            options=["Full Window (Jan 2024 – Oct 2026)", "Policy Zoom (Jul 2025 – Oct 2026)"],
            index=0,
        )

    pair_df = df_all[df_all["Region"].isin([REGION_A, peer_region])].copy()
    pair_df["Treated_Post"] = pair_df["Treated_Region_A"] * pair_df["Post"]
    did_mod = smf.ols("Expected_Pullback_Pct ~ Post + Treated_Region_A + Treated_Post", data=pair_df).fit(cov_type="HC1")

    pre_pair = pair_df[pair_df["Post"] == 0].copy()
    pre_pair["RegA_x_Day"] = pre_pair["Treated_Region_A"] * pre_pair["Day"]
    pt_mod = smf.ols("Expected_Pullback_Pct ~ Treated_Region_A + Day + RegA_x_Day", data=pre_pair).fit(cov_type="HC1")
    pt_pval = float(pt_mod.pvalues["RegA_x_Day"])

    att = float(did_mod.params["Treated_Post"])
    att_p = float(did_mod.pvalues["Treated_Post"])
    att_p_txt = "p < 0.001" if att_p < 0.001 else f"p = {att_p:.3f}"
    peer_short = peer_region.split(":")[0]

    if pt_pval > 0.05:
        st.success(
            f"**The Upshot:** Comparing **Region A (Germany)** against **{peer_region}**, the January 1, 2026 restaurant VAT cut "
            f"is associated with a **{abs(att):.2f} percentage-point** reduction in expected restaurant pullback ({att_p_txt}). "
            f"Pre-2026 trend slopes do not differ significantly (**p = {pt_pval:.4f}**), which is consistent with parallel trends "
            f"(they hold by construction in this simulation, so this check illustrates the method rather than proving anything about the real world)."
        )
    else:
        st.warning(
            f"**Caution on Parallel Trends:** Against **{peer_region}**, the naive DiD estimate is **{att:+.2f} pp**, "
            f"but the pre-2026 slopes diverge significantly (**p = {pt_pval:.4f}**), so this comparison is not credible. "
            f"This is why **Region B (Benelux Border Metro)** is the cleaner benchmark here (or why a Synthetic Control "
            f"would be needed if no single peer tracked Germany closely)."
        )

    fig2, ax2 = plt.subplots(figsize=(10.5, 5.0), dpi=180)
    ax2.set_facecolor("white")
    fig2.patch.set_facecolor("white")

    for reg in [REGION_A, peer_region]:
        sub = pair_df[pair_df["Region"] == reg]
        ax2.scatter(
            sub["Date"],
            sub["Expected_Pullback_Pct"],
            s=7,
            alpha=0.45,
            color=REGION_COLORS[reg],
            edgecolors="none",
            label=reg,
        )

    for reg, line_col in [(REGION_A, "#8B0000"), (peer_region, "#1F2937")]:
        for post_val, lbl_suffix in [(0, "Trend"), (1, None)]:
            sub_p = pair_df[(pair_df["Region"] == reg) & (pair_df["Post"] == post_val)]
            coef = np.polyfit(sub_p["Day"], sub_p["Expected_Pullback_Pct"], 1)
            x_days = np.array([0, POLICY_DAY - 1]) if post_val == 0 else np.array([POLICY_DAY, N_DAYS])
            x_dates = START_DATE + pd.to_timedelta(x_days, unit="D")
            ax2.plot(
                x_dates,
                coef[0] * x_days + coef[1],
                color=line_col,
                linewidth=2.4,
                label=f"{reg.split(':')[0]} Fitted Trend" if lbl_suffix else None,
            )

    ax2.axvline(POLICY_DATE, color="#444444", linestyle="--", linewidth=1.4, label="Restaurant Food VAT Cut: 19% → 7% (Region A Only)")
    ax2.set_title(f"Difference-in-Differences Evaluation: Germany (Region A) vs. {peer_short}", fontsize=12.5, pad=12)
    ax2.set_xlabel("Date", fontsize=10.5)
    ax2.set_ylabel("Expected Dining-Out Pullback Rate (%)*", fontsize=10.5)

    if zoom_window.startswith("Policy Zoom"):
        ax2.set_xlim(pd.Timestamp("2025-07-01"), pd.Timestamp("2026-12-01"))
    else:
        ax2.set_xlim(pd.Timestamp("2023-12-15"), pd.Timestamp("2026-12-01"))

    y_min = max(9.5, float(pair_df["Expected_Pullback_Pct"].min()) - 0.6)
    y_max = float(pair_df["Expected_Pullback_Pct"].max()) + 1.8
    ax2.set_ylim(y_min, y_max)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.grid(True, linestyle="-", linewidth=0.4, color="#E5E5E5")
    ax2.set_axisbelow(True)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#444444")
        spine.set_linewidth(0.8)

    ax2.legend(loc="upper right", ncol=2, frameon=True, facecolor="white", edgecolor="#DDDDDD", fontsize=8.2, markerscale=2.0)
    plt.tight_layout()
    st.pyplot(fig2, width="stretch")
    plt.close(fig2)

    st.markdown(
        "Use the comparison dropdown or time-window zoom above to test whether restaurant diners in Germany broke away from "
        "regional drift once the 7% restaurant-food VAT took effect on January 1, 2026."
    )

    st.divider()
    did_table = pd.DataFrame(
        {
            "Term": [
                f"Intercept ({peer_short} Pre-2026)",
                "Post (Jan 2026+ Tax Regime)",
                "Germany (Region A) Baseline Difference",
                "Estimated Effect of VAT Cut (Region A × Post)",
            ],
            "Estimate (pp)": [
                did_mod.params["Intercept"],
                did_mod.params["Post"],
                did_mod.params["Treated_Region_A"],
                did_mod.params["Treated_Post"],
            ],
            "Std. Error": [
                did_mod.bse["Intercept"],
                did_mod.bse["Post"],
                did_mod.bse["Treated_Region_A"],
                did_mod.bse["Treated_Post"],
            ],
            "p-value": [
                did_mod.pvalues["Intercept"],
                did_mod.pvalues["Post"],
                did_mod.pvalues["Treated_Region_A"],
                did_mod.pvalues["Treated_Post"],
            ],
        }
    )
    st.dataframe(
        did_table.style.format({"Estimate (pp)": "{:.3f}", "Std. Error": "{:.3f}", "p-value": "{:.4f}"}),
        width="stretch",
        hide_index=True,
    )

    st.markdown(
        """
        <div class="footnote-box">
            <strong>* Difference-in-Differences Specification &amp; Alternative Quasi-Experimental Designs:</strong><br>
            The baseline DiD specification estimates:
            <span class="equation-line">Expected Dining-Out Pullback Rate (%) = α + β·Post + δ·RegionA + θ·(RegionA × Post) + ε</span>
            Standard errors are heteroskedasticity-robust (HC1) and treat each simulated diner as independent; with real data, errors would need to be
            clustered, and with only two regions, cluster-robust inference is itself unreliable.
            If no single neighboring market had matched Germany's pre-2026 dining trend cleanly, a <strong>Synthetic Control</strong> weighting
            multiple European donor markets would construct the counterfactual; if a VAT rebate or voucher had been assigned via a sharp
            household spending or inactivity threshold, a <strong>Regression Discontinuity Design (RDD)</strong> would evaluate the cutoff directly.
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------------------------------------------------
# VIEW 3: CAUSAL FOREST
# -------------------------------------------------------------------------
else:
    if CausalForestDML is None:
        st.error("This view needs the `econml` package. Run `pip install -r requirements.txt` and restart the app.")
        st.stop()

    c1, c2 = st.columns([1.8, 1.5], vertical_alignment="center")
    with c1:
        max_gap_filter = st.slider(
            "Maximum days since last restaurant visit included in evaluation sample",
            min_value=15,
            max_value=90,
            value=90,
            step=5,
        )
    with c2:
        show_overall_avg = st.toggle("Show overall DiD reference line", value=True)

    ab_df = df_all[df_all["Region"].isin([REGION_A, REGION_B])].copy()
    ab_df["Treated_Post"] = ab_df["Treated_Region_A"] * ab_df["Post"]
    did_ref = float(
        smf.ols("Expected_Pullback_Pct ~ Post + Treated_Region_A + Treated_Post", data=ab_df)
        .fit(cov_type="HC1")
        .params["Treated_Post"]
    )

    seg_df = fit_causal_forest(int(max_gap_filter))

    top_seg = seg_df.iloc[0]
    bot_seg = seg_df.iloc[-1]
    st.success(
        f"**The Upshot:** Within the selected dining-interval window (up to **{max_gap_filter} days**), "
        f"**{top_seg['Diner_Segment']}** show the largest estimated drop in pullback (**{top_seg['Estimated_pp']:.2f} pp**; "
        f"simulation truth {top_seg['True_Effect_pp']:.2f} pp), whereas **{bot_seg['Diner_Segment']}** shift by only "
        f"**{bot_seg['Estimated_pp']:.2f} pp** (truth {bot_seg['True_Effect_pp']:.2f} pp). The forest recovers the heterogeneity "
        f"built into the simulation, with its intervals shown below."
    )

    fig3, ax3 = plt.subplots(figsize=(10.5, 4.5), dpi=180)
    ax3.set_facecolor("white")
    fig3.patch.set_facecolor("white")

    y_pos = np.arange(len(seg_df))
    vals = seg_df["Estimated_pp"].values
    errs = 1.96 * seg_df["SE"].values
    bar_cols = ["#B91C1C" if v < -2.5 else ("#DC2626" if v < -1.5 else "#9CA3AF") for v in vals]

    ax3.barh(y_pos, vals, height=0.52, color=bar_cols, edgecolor="#374151", linewidth=0.7, label="Estimated effect (95% CI)")
    ax3.errorbar(vals, y_pos, xerr=errs, fmt="none", ecolor="#111827", elinewidth=1.3, capsize=4)
    ax3.scatter(
        seg_df["True_Effect_pp"].values,
        y_pos,
        marker="D",
        s=55,
        color="white",
        edgecolors="#111827",
        linewidths=1.4,
        zorder=5,
        label="Simulation truth",
    )
    ax3.axvline(0, color="#444444", linewidth=0.9)
    if show_overall_avg:
        ax3.axvline(did_ref, color="#111827", linestyle="--", linewidth=1.2, label=f"Overall DiD estimate ({did_ref:.2f} pp)")

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(seg_df["Diner_Segment"], fontsize=9.5)
    ax3.set_xlabel("Estimated Change in Expected Dining-Out Pullback Rate (pp)", fontsize=10)
    ax3.set_title("Honest Causal Forest: Which Diners Responded to Germany's Restaurant-Food VAT Cut?", fontsize=12, pad=12)
    ax3.set_xlim(-4.8, 0.8)
    ax3.grid(True, axis="x", linestyle="-", linewidth=0.4, color="#E5E5E5")
    ax3.set_axisbelow(True)
    for spine in ax3.spines.values():
        spine.set_edgecolor("#444444")
        spine.set_linewidth(0.8)
    ax3.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#DDDDDD", fontsize=8.6)
    plt.tight_layout()
    st.pyplot(fig3, width="stretch")
    plt.close(fig3)

    st.markdown(
        "Adjust the dining-interval slider above to test how the estimates change when excluding long-lapsed diners."
    )

    st.divider()
    table = seg_df[
        [
            "Diner_Segment",
            "Diners",
            "Avg_Days_Since_Dining",
            "Avg_Venues_Visited",
            "Estimated_pp",
            "CI_Low",
            "CI_High",
            "True_Effect_pp",
        ]
    ].rename(
        columns={
            "Diner_Segment": "Diner Segment",
            "Avg_Days_Since_Dining": "Avg Days Since Dining",
            "Avg_Venues_Visited": "Avg Venues",
            "Estimated_pp": "Estimated Effect (pp)",
            "CI_Low": "95% CI Low",
            "CI_High": "95% CI High",
            "True_Effect_pp": "Simulation Truth (pp)",
        }
    )
    st.dataframe(
        table.style.format(
            {
                "Diners": "{:,}",
                "Avg Days Since Dining": "{:.1f} days",
                "Avg Venues": "{:.2f}",
                "Estimated Effect (pp)": "{:.2f}",
                "95% CI Low": "{:.2f}",
                "95% CI High": "{:.2f}",
                "Simulation Truth (pp)": "{:.2f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.markdown(
        """
        <div class="footnote-box">
            <strong>* Method:</strong><br>
            An honest causal forest (EconML <code>CausalForestDML</code>) estimates how the effect of the cut varies with diner characteristics:
            <span class="equation-line">CATE(x) = E[Y(1) − Y(0) | Diner Profile X = x]</span>
            "Honest" means each tree uses one subsample to choose its splits and a separate subsample to estimate the effects in its leaves.
            The model also residualizes the outcome and treatment with random-forest nuisance models (double machine learning), and it is
            <strong>cross-fitted</strong> in two folds, so every diner's effect is estimated by a forest that never saw that diner.
            Outcomes are baseline-adjusted (each diner's pullback rate minus their region's pre-2026 mean), so the forest's average is comparable to the DiD estimate
            under parallel trends. Segment intervals combine fold-level standard errors and are approximate.
            <strong>Caveats:</strong> treatment is assigned by region rather than randomly, so the forest relies on parallel trends plus the assumption
            that diner profiles capture all relevant differences between regions; the simulation builds both in. "Simulation truth" is the known effect used to generate the data.
            All figures use seeded synthetic records (<code>seed = 42</code>).
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------------------------------------------------
# ASSUMPTIONS & LIMITATIONS (all views)
# -------------------------------------------------------------------------
with st.expander("Assumptions & limitations"):
    st.markdown(
        """
- **Simulated data.** Every record is generated by code (`seed = 42`). Nothing here is real transaction data, and the results illustrate a method; they do not measure the actual effect of Germany's VAT cut. This is an independent portfolio project with simulated data.
- **Parallel trends hold by construction** for Region A and Region B, so the pre-trend test passing is expected. Real data would need to earn that result.
- **Possible spillovers.** A border-metro comparison group could be contaminated if diners cross the border to eat in Germany after the cut.
- **Pass-through is assumed.** The simulation assumes the tax saving reaches diners. Real-world evidence suggests restaurants have passed on little of it in menu prices, which would shrink the effect on behavior.
- **The cut covers restaurant food only.** Beverages stay at the 19% rate.
- **Inference is simplified.** Standard errors treat each simulated diner as independent; real analyses would cluster and would still face few-cluster problems with two regions.
- **Repeated cross-section, not a panel.** Each row is a different simulated diner observed on one date.
        """
    )
