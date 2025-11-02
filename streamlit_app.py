# streamlit_app.py
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

# ---------------------------
# Page Config & Header
# ---------------------------
st.set_page_config(page_title="Customer Churn — Survival & Profitability Dashboard", layout="wide")
st.title("📊 Customer Churn — Survival, Profitability & Strategy Dashboard")
st.caption("Developed by **Sohan Ghosh | MSc Data Science & AI | University of Calcutta**")

with st.sidebar:
    st.header("🧭 Navigation")
    st.info("Use this dashboard to explore survival modeling, retention strategy, and profitability simulation.")
    st.markdown("---")
    st.caption("Upload model (.pkl) & processed CSV → Compute churn risk → Simulate offer profit → Download targets")

# ---------------------------
# Upload section
# ---------------------------
col1, col2 = st.columns(2)
with col1:
    model_file = st.file_uploader("Upload RSF model (.pkl)", type=["pkl"])
with col2:
    data_file = st.file_uploader("Upload processed churn dataset (.csv)", type=["csv"])

H = st.slider("Select churn horizon (months)", 3, 24, 10, 1)

# ---------------------------
# Helper functions
# ---------------------------
def safe_features(model, df):
    if hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)
    else:
        features = [c for c in df.columns if c not in ["duration", "event", "Exited"]]
    matched = [c for c in features if c in df.columns]
    return matched, features

def compute_churn(rsf, df, horizon):
    matched, model_features = safe_features(rsf, df)
    if not matched:
        raise ValueError("No overlapping columns found between data and model.")
    X = df[matched]
    times = np.asarray(rsf.unique_times_, dtype=float)
    idx = np.searchsorted(times, horizon, side="right") - 1
    idx = max(0, min(idx, len(times) - 1))
    surv = rsf.predict_survival_function(X, return_array=True)
    if not isinstance(surv, np.ndarray):
        surv = np.vstack([sf(times) for sf in rsf.predict_survival_function(X)])
    p_churn = 1 - surv[:, idx]
    return p_churn, float(times[idx]), matched, model_features

def npv_calc(df, rev, cost, uplift, months=12):
    uplift = uplift / 100
    p0 = df["p_churn_H"].clip(0, 1)
    p1 = (p0 * (1 - uplift)).clip(0, 1)
    df["INV"] = (p0 - p1) * rev * months - cost
    return df

# ---------------------------
# Main Flow
# ---------------------------
if data_file is not None:
    df = pd.read_csv(data_file)
    st.success(f"✅ Loaded dataset with {len(df):,} rows and {len(df.columns)} columns.")
    st.dataframe(df.head(10), use_container_width=True)

    # Check for churn column
    if "p_churn_H" not in df.columns:
        if model_file is None:
            st.error("⚠️ No `p_churn_H` column found — please upload your RSF `.pkl` model to compute churn probabilities.")
        else:
            try:
                rsf = joblib.load(model_file)
                p_churn, used_h, matched, model_feats = compute_churn(rsf, df, H)

                st.markdown("### 🔍 Feature Matching Summary")
                st.write(f"**Matched columns:** {len(matched)} / {len(model_feats)}")
                st.json(matched)
                if len(matched) < len(model_feats):
                    st.warning("⚠️ Some model features not found in dataset. Predictions use matched subset only.")

                df["p_churn_H"] = p_churn
                st.success(f"✅ Computed churn probabilities using horizon {used_h:.2f} months.")
            except Exception as e:
                st.error(f"❌ Failed to compute churn probabilities: {e}")
    else:
        st.info("Found existing `p_churn_H` in dataset — skipping model prediction.")

    # ---------------------------
    # Visualization & NPV
    # ---------------------------
    if "p_churn_H" in df.columns:
        st.markdown("---")
        st.header("📈 Survival Probability & Churn Distribution")

        if "duration" in df.columns and "event" in df.columns:
            km = KaplanMeierFitter()
            km.fit(df["duration"], event_observed=df["event"])
            fig, ax = plt.subplots()
            km.plot_survival_function(ax=ax)
            ax.set_title("Kaplan–Meier Survival Curve")
            ax.set_xlabel("Time (months)")
            ax.set_ylabel("Survival Probability")
            st.pyplot(fig)
        else:
            st.info("No duration/event columns found — showing churn probability distribution instead.")

        st.bar_chart(df["p_churn_H"].sample(min(len(df), 100)))
        st.metric("Average churn probability", f"{df['p_churn_H'].mean():.3f}")

        # ---------------------------
        # Offer Strategy
        # ---------------------------
        st.markdown("---")
        st.header("💰 Offer Strategy — Net Present Value (NPV) Simulation")

        c1, c2, c3 = st.columns(3)
        rev = c1.number_input("Avg Monthly Revenue per Customer (₹)", 100.0, 5000.0, 500.0)
        cost = c2.number_input("Offer Cost per Customer (₹)", 0.0, 2000.0, 200.0)
        uplift = c3.slider("Retention Uplift (%)", 0, 100, 25)

        df = npv_calc(df, rev, cost, uplift)
        prof = df[df["INV"] > 0]

        cA, cB, cC, cD = st.columns(4)
        cA.metric("Customers", f"{len(df):,}")
        cB.metric("Profitable Targets", f"{len(prof):,}")
        cC.metric("Mean INV (₹)", f"{df['INV'].mean():.2f}")
        cD.metric("Total Profit (₹)", f"{prof['INV'].sum():.2f}")

        st.dataframe(df[["p_churn_H", "INV"]].join(df.drop(columns=["p_churn_H", "INV"], errors="ignore")).head(20),
                     use_container_width=True)

        st.download_button(
            "⬇️ Download Full Predictions",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="churn_predictions.csv",
            mime="text/csv"
        )

        st.download_button(
            "⬇️ Download Target List (Profitable Only)",
            data=prof.to_csv(index=False).encode("utf-8"),
            file_name="target_list_RSF_NPV.csv",
            mime="text/csv"
        )

        st.success("✅ Analysis complete. Use the sliders to simulate profit sensitivity.")

else:
    st.info("📤 Upload your processed churn dataset (.csv) to begin.")
