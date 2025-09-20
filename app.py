# ---------- Imports ----------
import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import mean_absolute_error, accuracy_score, precision_score, recall_score

# ---------- Page config (must be first Streamlit call) ----------
st.set_page_config(
    page_title="ML for Community Microgrids - CYM 2025",
    page_icon="⚡",  # can be an emoji
    layout="wide",
)

#--------- Header & Title --------------------------------------
st.title("Machine Learning for Community Microgrids")
st.subheader("Forecasting • Fault Detection • Community Focus")


# ---------- Minimal corporate CSS ----------
st.markdown(
    """
<style>
.block-container {padding-top: 2rem; padding-bottom: 2.5rem;}
h1, h2, h3 { color:#111827; letter-spacing:-0.01em; font-weight:800; }
p, li, .stMarkdown { color:#111827; }
.small-muted { color:#6B7280; font-size:0.9rem; }

/* metric cards */
.metric-wrap {
  border:1px solid #E5E7EB; border-radius:14px; background:#FFF;
  padding:16px 18px; box-shadow:0 1px 2px rgba(0,0,0,.04);
}

/* hide Streamlit chrome */
#MainMenu, header [data-testid="stToolbar"], footer {visibility:hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Tiny black SVG icons (no emoji) ----------
def icon_svg(name, size=22):
    svgs = {
        "bolt": (
            '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" '
            'xmlns="http://www.w3.org/2000/svg"><path d="M13 3L4 14h6l-1 7 9-11h-6l1-7z" '
            'stroke="#111827" stroke-width="1.6" fill="none" stroke-linejoin="round"/></svg>'
        ),
        "target": (
            '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" '
            'xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="8" '
            'stroke="#111827" stroke-width="1.6"/><circle cx="12" cy="12" r="3" '
            'fill="#111827"/></svg>'
        ),
        "shield": (
            '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" '
            'xmlns="http://www.w3.org/2000/svg"><path d="M12 3l8 3v6c0 4.5-3.1 8-8 9-4.9-1-8-4.5-8-9V6l8-3z" '
            'stroke="#111827" stroke-width="1.6" fill="none"/></svg>'
        ),
    }
    return svgs[name].format(s=size)


@st.cache_data
def load_csv(path, parse_dates=None):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=parse_dates)

# ---------- Tabs (corporate labels, no emojis) ----------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Forecasting",
        "Fault Detection",
        "Community Summary",
        "Household Explorer",
        "Model Performance",
        "Scheduling",  # <-- NEW
    ]
)

# ===================== 1) Forecasting =====================
with tab1:
    st.subheader("Forecasting — Actual vs Baseline (Linear) vs Neural")

    # A) Prefer Colab outputs if present
    fe_path = "data/forecast_eval.csv"
    if os.path.exists(fe_path):
        fe = pd.read_csv(fe_path)

        # Standardize column names
        colmap = {
            "t": "Time",
            "time": "Time",
            "timestamp": "Time",
            "y_true": "Actual",
            "actual": "Actual",
            "y": "Actual",
            "y_pred_linear": "Linear",
            "linear": "Linear",
            "y_pred_neural": "Neural",
            "neural": "Neural",
        }
        for c in list(fe.columns):
            if c in colmap and colmap[c] not in fe.columns:
                fe.rename(columns={c: colmap[c]}, inplace=True)

        need = {"Actual", "Linear", "Neural"}
        if need.issubset(set(fe.columns)):
            # -------- NEW: keep raw copy for honest metrics & residuals --------
            fe_raw = fe.copy()

            # KPIs from raw (unsmoothed) data
            mae_lin = mean_absolute_error(fe_raw["Actual"], fe_raw["Linear"])
            mae_nn = mean_absolute_error(fe_raw["Actual"], fe_raw["Neural"])
            imp_pct = 100.0 * (mae_lin - mae_nn) / max(mae_lin, 1e-9)

            k1, k2, k3 = st.columns(3)
            k1.metric("Baseline (Linear) MAE", f"{mae_lin:.3f}")
            k2.metric("Neural Net MAE", f"{mae_nn:.3f}", f"−{imp_pct:.2f}%")
            k3.caption("Walk-forward validation; lower MAE is better.")

            # -------- NEW: gentle display-only cleanup --------
            fe_display = fe.copy()

            # clip impossible negatives for display (does NOT affect metrics)
            for _c in ["Linear", "Neural"]:
                if _c in fe_display.columns:
                    fe_display[_c] = fe_display[_c].clip(lower=0)

            # optional light smoothing for visual clarity (does NOT affect metrics)
            for _c in ["Actual", "Linear", "Neural"]:
                if _c in fe_display.columns:
                    fe_display[_c] = fe_display[_c].rolling(5, min_periods=1).mean()

            # -------- NEW: friendlier x-axis --------
            xcol = "Time" if "Time" in fe_display.columns else fe_display.index.name or "index"
            if xcol == "index":
                fe_display = fe_display.reset_index().rename(columns={"index": "index"})
                xcol = "index"

            # If Time is numeric (e.g., 315..680), cast to "Hours since start"
            x_label = xcol
            try:
                # numeric-like?
                if np.issubdtype(fe_display[xcol].dtype, np.number):
                    # estimate step (fallback 15 min)
                    vals = fe_display[xcol].values
                    if len(vals) > 2:
                        step = np.median(np.diff(np.unique(vals)))
                        step_minutes = 15 if not np.isfinite(step) or step <= 0 else 15  # keep 15m default
                    else:
                        step_minutes = 15
                    hours_since = (fe_display[xcol] - fe_display[xcol].min()) * (step_minutes / 60.0)
                    fe_display["_hours_since_start"] = hours_since
                    xcol = "_hours_since_start"
                    x_label = "Hours since start"
            except Exception:
                pass

            # Overlay chart (display data)
            fig_f = px.line(
                fe_display,
                x=xcol,
                y=["Actual", "Linear", "Neural"],
                labels={"value": "kWh", "variable": "Series", xcol: x_label},
                title="Actual vs Predictions (Test Windows)",
            )
            st.plotly_chart(fig_f, use_container_width=True)

            # Residuals (from raw, unsmoothed data)
            fe_raw["res_linear"] = fe_raw["Actual"] - fe_raw["Linear"]
            fe_raw["res_neural"] = fe_raw["Actual"] - fe_raw["Neural"]
            fig_r = px.histogram(
                fe_raw.melt(value_vars=["res_linear", "res_neural"], var_name="Model", value_name="Residual"),
                x="Residual",
                color="Model",
                nbins=40,
                barmode="overlay",
                title="Residual Distribution (lower spread is better)",
            )
            st.plotly_chart(fig_r, use_container_width=True)
        else:
            st.error(f"`forecast_eval.csv` needs columns {sorted(list(need))}. Found: {list(fe.columns)}")
    else:
        st.info("Add **data/forecast_eval.csv** from Colab to show Actual vs Linear vs Neural.")

    # B) Optional scenario sandbox if you also include a solar forecast file
    df_solar = load_csv("data/forecast.csv", parse_dates=["timestamp"])
    if df_solar is not None:
        st.markdown("### Scenario: Cloudiness & Uncertainty (Solar)")
        with st.sidebar:
            st.markdown("### Forecast Controls")
            cloudiness = st.slider("Cloudiness factor for next 24h (%)", 60, 120, 100, 5)

        horizon_start = df_solar["timestamp"].max() - pd.Timedelta(hours=24)
        df_solar["phase"] = (df_solar["timestamp"] > horizon_start).map(
            {True: "Forecast (next 24h)", False: "History"}
        )
        df_solar.loc[df_solar["phase"] == "Forecast (next 24h)", "generation_kw"] *= (cloudiness / 100)

        band = df_solar[df_solar["phase"] == "Forecast (next 24h)"].copy()
        if not band.empty:
            band["lo"] = band["generation_kw"] * 0.85
            band["hi"] = band["generation_kw"] * 1.15

        fig = px.line(
            df_solar,
            x="timestamp",
            y="generation_kw",
            color="phase",
            title="Historical + Next 24h Solar Forecast (with Scenario & Uncertainty)",
        )
        if not band.empty:
            fig.add_scatter(x=band["timestamp"], y=band["hi"], mode="lines", name="Uncertainty hi", line=dict(width=0))
            fig.add_scatter(
                x=band["timestamp"],
                y=band["lo"],
                mode="lines",
                name="Uncertainty lo",
                fill="tonexty",
                line=dict(width=0),
            )
        fig.update_layout(legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    st.caption("**Result:** Neural network reduced mean absolute error by **13.27%** vs. a linear baseline.")

# ===================== 2) Fault Detection =====================
with tab2:
    st.subheader("Fault Detection — Random Forest Ensemble")

    # ---- A) Metrics from fault_eval.csv (Colab export)
    fd_path = "data/fault_eval.csv"
    y_true = y_pred = y_proba = None
    if os.path.exists(fd_path):
        fd = pd.read_csv(fd_path)
        # Standardize columns if they differ
        fcolmap = {
            "y_true": "y_true",
            "actual": "y_true",
            "y_pred_rf": "y_pred_rf",
            "predicted": "y_pred_rf",
            "y_proba_rf": "y_proba_rf",
            "proba": "y_proba_rf",
            "prob": "y_proba_rf",
        }
        for c in list(fd.columns):
            if c in fcolmap and fcolmap[c] not in fd.columns:
                fd.rename(columns={c: fcolmap[c]}, inplace=True)

        need_fd = {"y_true", "y_pred_rf"}
        if need_fd.issubset(set(fd.columns)):
            from sklearn.metrics import (
                accuracy_score,
                precision_score,
                recall_score,
                confusion_matrix,
                precision_recall_curve,
            )

            y_true = fd["y_true"].astype(int).values
            y_pred = fd["y_pred_rf"].astype(int).values
            y_proba = fd["y_proba_rf"].astype(float).values if "y_proba_rf" in fd.columns else None

            # Base metrics from provided predictions
            acc = accuracy_score(y_true, y_pred) * 100.0
            prec = precision_score(y_true, y_pred, zero_division=0) * 100.0
            rec = recall_score(y_true, y_pred, zero_division=0) * 100.0

            d1, d2, d3 = st.columns(3)
            d1.metric("Accuracy", f"{acc:.1f}%")
            d2.metric("Precision", f"{prec:.1f}%")
            d3.metric("Recall", f"{rec:.1f}%")

            # Confusion matrix (base)
            cm = confusion_matrix(y_true, y_pred)
            cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
            fig_cm = px.imshow(cm_df, text_auto=True, aspect="auto", title="Confusion Matrix (baseline threshold)")
            st.plotly_chart(fig_cm, use_container_width=True)

            # -------- NEW: Threshold tuning if probabilities are available --------
            if y_proba is not None:
                st.markdown("### Threshold Tuning")
                thr = st.slider("Classification threshold", 0.05, 0.95, 0.50, 0.01)
                y_pred_tuned = (y_proba >= thr).astype(int)

                acc_t = accuracy_score(y_true, y_pred_tuned) * 100.0
                prec_t = precision_score(y_true, y_pred_tuned, zero_division=0) * 100.0
                rec_t = recall_score(y_true, y_pred_tuned, zero_division=0) * 100.0

                t1, t2, t3 = st.columns(3)
                t1.metric("Accuracy (tuned)", f"{acc_t:.1f}%")
                t2.metric("Precision (tuned)", f"{prec_t:.1f}%")
                t3.metric("Recall (tuned)", f"{rec_t:.1f}%")

                cm_t = confusion_matrix(y_true, y_pred_tuned)
                cm_t_df = pd.DataFrame(cm_t, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
                fig_cm_t = px.imshow(
                    cm_t_df, text_auto=True, aspect="auto", title=f"Confusion Matrix @ threshold={thr:.2f}"
                )
                st.plotly_chart(fig_cm_t, use_container_width=True)

                # Precision–Recall Curve
                p, r, _ = precision_recall_curve(y_true, y_proba)
                pr_df = pd.DataFrame({"precision": p, "recall": r})
                fig_pr = px.line(pr_df, x="recall", y="precision", title="Precision–Recall Curve")
                st.plotly_chart(fig_pr, use_container_width=True)
        else:
            st.warning(f"`fault_eval.csv` missing required columns {sorted(list(need_fd))}. Found: {list(fd.columns)}")
    else:
        st.info("Optional: add **data/fault_eval.csv** to compute accuracy/precision/recall here.")

    st.caption("**Result:** Fault detector achieved **93.9% accuracy**, balancing sensitivity and precision.")
    st.divider()
    st.markdown("### Live Monitor — System Fault Indicator")

# Re-use the same faults.csv you already visualize (timestamp, generation_kw, fault)
fpath = "data/faults.csv"
if not os.path.exists(fpath):
    st.info("Add **data/faults.csv** (with columns: timestamp [optional], generation_kw, fault=0/1) to run the live demo.")
else:
    demo = pd.read_csv(fpath)

    # Be forgiving with column names
    colmap = {
        "time": "timestamp",
        "ts": "timestamp",
        "date": "timestamp",
        "kw": "generation_kw",
        "gen": "generation_kw",
        "output": "generation_kw",
        "power_kw": "generation_kw",
        "label": "fault",
        "anomaly": "fault",
        "is_fault": "fault",
    }
    for c in list(demo.columns):
        if c in colmap and colmap[c] not in demo.columns:
            demo.rename(columns={c: colmap[c]}, inplace=True)

    if "fault" not in demo.columns:
        st.error("`faults.csv` must include a **fault** column (0/1).")
    else:
        # Sort by time/index for deterministic playback
        if "timestamp" in demo.columns:
            try:
                demo["timestamp"] = pd.to_datetime(demo["timestamp"])
                demo = demo.sort_values("timestamp")
            except Exception:
                demo = demo.reset_index(drop=True)
        else:
            demo = demo.reset_index(drop=True)

        # --- Optional: load forecast to enable preventive (amber) warnings ---
        fc_path = "data/forecast.csv"
        forecast_df = load_csv(fc_path, parse_dates=["timestamp"])
        if forecast_df is not None and {"timestamp", "generation_kw"}.issubset(forecast_df.columns):
            forecast_df = forecast_df.rename(columns={"generation_kw": "forecast_kw"})
            try:
                forecast_df = forecast_df.sort_values("timestamp")
            except Exception:
                forecast_df = None
        else:
            forecast_df = None


        # --- Controls ---
        left, right = st.columns([1, 2])
        with left:
            speed = st.select_slider(
                "Playback speed",
                options=[0.25, 0.5, 1.0, 2.0, 4.0],
                value=1.0,
                help="x realtime (demo)",
            )
            window = st.slider("Window size (points)", 30, min(300, max(60, len(demo))), 120, 10)
        with right:
            run = st.toggle("▶ Play demo", value=False, help="When ON, the indicator updates as we stream the file")

        # --- CSS for the big indicator ---
        st.markdown(
            """
<style>
.status-wrap {display:flex;align-items:center;gap:14px;margin:10px 0 6px;}
.status-dot {width:30px;height:30px;border-radius:50%;box-shadow:0 0 10px rgba(16,185,129,.6) inset, 0 0 14px rgba(16,185,129,.35);}
.ok {background:#10B981;}
.amber {background:#F59E0B; box-shadow:0 0 10px rgba(245,158,11,.65) inset, 0 0 16px rgba(245,158,11,.45); animation:pulseAmber 1.1s ease-in-out infinite;}
.fault {background:#EF4444;box-shadow:0 0 10px rgba(239,68,68,.7) inset, 0 0 16px rgba(239,68,68,.55); animation:pulse 1.1s ease-in-out infinite;}
@keyframes pulse {
  0%   {transform:scale(1);   box-shadow:0 0 10px rgba(239,68,68,.7) inset, 0 0 16px rgba(239,68,68,.55);}
  50%  {transform:scale(1.08);box-shadow:0 0 14px rgba(239,68,68,.85) inset, 0 0 22px rgba(239,68,68,.7);}
  100% {transform:scale(1);   box-shadow:0 0 10px rgba(239,68,68,.7) inset, 0 0 16px rgba(239,68,68,.55);}
}
@keyframes pulseAmber {
  0%   {transform:scale(1);   box-shadow:0 0 10px rgba(245,158,11,.65) inset, 0 0 16px rgba(245,158,11,.45);}
  50%  {transform:scale(1.08);box-shadow:0 0 14px rgba(245,158,11,.85) inset, 0 0 22px rgba(245,158,11,.65);}
  100% {transform:scale(1);   box-shadow:0 0 10px rgba(245,158,11,.65) inset, 0 0 16px rgba(245,158,11,.45);}
}
.status-text {font-weight:700;letter-spacing:.2px;}
.muted {color:#6B7280;font-size:0.92rem;margin-top:-4px;}
</style>
""",
            unsafe_allow_html=True,
        )

        # === NEW: simple imminent-drop heuristic (proactive "amber") ===
                # === Preventive risk heuristic (amber): drift + trend + forecast deviation ===
        def preventive_risk(sub_df: pd.DataFrame, forecast_df: pd.DataFrame | None) -> bool:
            """
            Amber if EITHER:
              A) (Drift + Trend)  -> steep short-term negative drift AND short MA below long MA, OR
              B) (Forecast miss)  -> actual underperforms forecast by > threshold for several recent points.

            Tunable thresholds below are chosen to be reasonably conservative for a demo.
            """
            if "generation_kw" not in sub_df.columns or len(sub_df) < 12:
                return False

            g = sub_df["generation_kw"].astype(float).values
            n = len(g)

            # ---- A) Drift + Trend
            # Use last 8 points for drift, last 4/12 for MA crossover
            look_drift = min(8, n - 1)
            base = max(g[-look_drift-1], 1e-6)
            slope_rel = (g[-1] - g[-look_drift-1]) / base     # relative change
            ma_short = np.mean(g[-4:]) if n >= 4 else np.mean(g[-n:])
            ma_long  = np.mean(g[-12:]) if n >= 12 else np.mean(g[-n:])
            trend_crossover = ma_short < (ma_long * 0.97)      # 3% below long MA
            drift_trigger   = slope_rel < -0.05                 # ≥5% drop over the window

            drift_and_trend = drift_trigger and trend_crossover

            # ---- B) Forecast deviation (if available + timestamp present)
            forecast_trigger = False
            if forecast_df is not None and "timestamp" in sub_df.columns:
                try:
                    # align last ~6 points with nearest forecast (<= 10 minutes tolerance)
                    tail = sub_df[["timestamp", "generation_kw"]].tail(6).copy()
                    tail = tail.sort_values("timestamp")
                    f_aligned = pd.merge_asof(
                        tail.rename(columns={"timestamp": "ts_actual"}).sort_values("ts_actual"),
                        forecast_df.rename(columns={"timestamp": "ts_forecast"}).sort_values("ts_forecast"),
                        left_on="ts_actual", right_on="ts_forecast", direction="nearest", tolerance=pd.Timedelta("10min")
                    )
                    # if forecast match exists, compute relative error (actual vs forecast)
                    if "forecast_kw" in f_aligned.columns:
                        f_aligned["rel_err"] = (f_aligned["generation_kw"] - f_aligned["forecast_kw"]) / f_aligned["forecast_kw"].clip(lower=1e-6)
                        # Trigger if at least half of the aligned recent points underperform forecast by >20%
                        underperf = (f_aligned["rel_err"] < -0.20).sum()
                        total_aligned = f_aligned["forecast_kw"].notna().sum()
                        forecast_trigger = (total_aligned >= 3) and (underperf >= max(2, total_aligned // 2))
                except Exception:
                    forecast_trigger = False

            return bool(drift_and_trend or forecast_trigger)


        # Keep our place while "playing"
        if "monitor_idx" not in st.session_state:
            st.session_state.monitor_idx = 0

        placeholder = st.empty()
        chartspot = st.empty()

        def render_step(i: int):
            row = demo.iloc[i] 
            is_fault = int(row["fault"]) == 1
            stamp = row["timestamp"] if "timestamp" in demo.columns else i

            # Tiny rolling window so judges see where we are in the series
            start = max(0, i - window + 1)
            sub = demo.iloc[start : i + 1]

            # Decide status: RED (fault) > AMBER (imminent drop) > GREEN (ok)
            if is_fault:
                label = "🚨 Fault detected"
                cls = "fault"
            else:
                amber = preventive_risk(sub, forecast_df)
                if amber:
                    label = "⚠️ Imminent drop likely"
                    cls = "amber"
                else:
                    label = "✅ System healthy"
                    cls = "ok"

            placeholder.markdown(
                f'<div class="status-wrap"><div class="status-dot {cls}"></div>'
                f'<div class="status-text">{label}</div></div>'
                f'<div class="muted">Now at: <b>{stamp}</b></div>',
                unsafe_allow_html=True,
            )

            # Rolling chart
            xcol = "timestamp" if "timestamp" in sub.columns else sub.index.name or "index"
            if xcol == "index":
                sub = sub.reset_index().rename(columns={"index": "index"})
                xcol = "index"
            fig_demo = px.line(sub, x=xcol, y="generation_kw", title="Live stream (demo)")
            faults_sub = sub[sub["fault"] == 1]
            if not faults_sub.empty:
                fig_demo.add_scatter(
                    x=faults_sub[xcol],
                    y=faults_sub["generation_kw"],
                    mode="markers",
                    marker=dict(color="red", size=9),
                    name="Fault",
                )
            fig_demo.update_layout(showlegend=True, legend_title_text="")
            chartspot.plotly_chart(fig_demo, use_container_width=True)

        # --- Play / step logic ---
        if run:
            import time

            # play from current index to end
            for i in range(st.session_state.monitor_idx, len(demo)):
                st.session_state.monitor_idx = i
                render_step(i)
                time.sleep(max(0.05, 0.25 / float(speed)))  # faster speed -> shorter sleep
        else:
            # paused – render current frame
            render_step(st.session_state.monitor_idx)

        # Quick step controls
        btns = st.columns(3)
        if btns[0].button("⏮ Restart"):
            st.session_state.monitor_idx = 0
            st.rerun()
        if btns[1].button("⏪ Step -1"):
            st.session_state.monitor_idx = max(0, st.session_state.monitor_idx - 1)
            st.rerun()
        if btns[2].button("⏩ Step +1"):
            st.session_state.monitor_idx = min(len(demo) - 1, st.session_state.monitor_idx + 1)
            st.rerun()

# ---- B) Time-series visualization from faults.csv (line + red markers)
fpath = "data/faults.csv"
if os.path.exists(fpath):
    df2 = pd.read_csv(fpath)

    # Be forgiving about column names
    colmap = {
        "time": "timestamp",
        "ts": "timestamp",
        "date": "timestamp",
        "kw": "generation_kw",
        "gen": "generation_kw",
        "output": "generation_kw",
        "power_kw": "generation_kw",
        "label": "fault",
        "anomaly": "fault",
        "is_fault": "fault",
    }
    for c in list(df2.columns):
        if c in colmap and colmap[c] not in df2.columns:
            df2.rename(columns={c: colmap[c]}, inplace=True)

    # Timestamp handling (optional)
    if "timestamp" in df2.columns:
        try:
            df2["timestamp"] = pd.to_datetime(df2["timestamp"])
        except Exception:
            pass
        x_axis = "timestamp"
    else:
        df2["index"] = np.arange(len(df2))
        x_axis = "index"

    # Required y & fault columns
    if "generation_kw" not in df2.columns or "fault" not in df2.columns:
        st.error("`faults.csv` needs at least: generation_kw and fault (0/1). Optional: timestamp.")
    else:
        df2["fault"] = df2["fault"].astype(int)
        df2 = df2.sort_values(x_axis)

        fig2 = px.line(df2, x=x_axis, y="generation_kw", title="Signal with Fault Flags")
        faults = df2[df2["fault"] == 1]
        if not faults.empty:
            fig2.add_scatter(
                x=faults[x_axis],
                y=faults["generation_kw"],
                mode="markers",
                marker=dict(color="red", size=9),
                name="Fault",
            )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Tip: `faults.csv` schema → **timestamp (optional), generation_kw, fault (0/1)**.")
else:
    st.info("Add **data/faults.csv** to see the time series with red fault markers.")

    # ==== Live Monitor: big red/green indicator (drop this at the END of tab2) ====

# ===================== 3) Community Summary =====================
with tab3:
    st.subheader("Community Impact (50 households)")

    agg_path = "data/community_agg.csv"
    im_path = "data/impact_metrics.json"

    agg = load_csv(agg_path, parse_dates=["timestamp"])
    if agg is not None:
        c1, c2, c3 = st.columns(3)
        if "avg_generation_kw" in agg.columns:
            c1.metric("Avg Generation (kW)", f"{agg['avg_generation_kw'].mean():.2f}")
        if "total_generation_kw" in agg.columns:
            c2.metric("Peak Community Output (kW)", f"{agg['total_generation_kw'].max():.1f}")
        if "households_faulty" in agg.columns:
            c3.metric("Faulty Homes (latest)", f"{int(agg.iloc[-1]['households_faulty'])}")

        ycols = [c for c in ["total_generation_kw", "avg_generation_kw"] if c in agg.columns]
        if ycols:
            fig3 = px.line(
                agg,
                x="timestamp",
                y=ycols,
                labels={"value": "kW", "variable": "Metric"},
                title="Community Output (Total & Average)",
            )
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Optionally add **data/community_agg.csv** to show community-level KPIs and trends.")

    # Savings calculator
    if os.path.exists(im_path):
        with open(im_path) as f:
            impact = json.load(f)
        st.markdown("### Savings Estimator")
        price = st.slider("Electricity price (¢/kWh)", 5, 40, 18)
        kwh = float(impact.get("energy_loss_kwh", 0.0))
        monthly_savings = kwh * (price / 100)
        annual_savings = monthly_savings * 12

        c1, c2, c3 = st.columns(3)
        c1.metric("Avoided Loss (kWh / demo window)", f"{kwh:.2f}")
        c2.metric("Estimated Savings (this window)", f"${monthly_savings:,.2f}")
        c3.metric("Annualized (if recurring)", f"${annual_savings:,.0f}")
        st.caption("Student housing & suburban neighborhoods serve as representative microgrid case studies.")

# ===================== 4) Household Explorer =====================
with tab4:
    st.subheader("Household Explorer")
    hh_path = "data/community_households.csv"
    agg_path = "data/community_agg.csv"

    if not (os.path.exists(hh_path) and os.path.exists(agg_path)):
        st.info(
            "Optionally add **data/community_households.csv** and **data/community_agg.csv** "
            "to explore households vs. community."
        )
    else:
        ids = pd.read_csv(hh_path, usecols=["household_id"]).drop_duplicates()["household_id"].tolist()
        if ids:
            sel = st.selectbox("Select a household", options=ids, index=0)
            hh = load_csv(hh_path, parse_dates=["timestamp"])
            hh_sel = hh[hh["household_id"] == sel]
            agg_small = load_csv(agg_path, parse_dates=["timestamp"])

            c1, c2 = st.columns(2)
            with c1:
                fig_hh = px.line(hh_sel, x="timestamp", y="generation_kw", title=f"{sel} Output (kW)")
                st.plotly_chart(fig_hh, use_container_width=True)
            with c2:
                if "avg_generation_kw" in agg_small.columns:
                    fig_avg = px.line(agg_small, x="timestamp", y="avg_generation_kw", title="Community Average Output (kW)")
                    st.plotly_chart(fig_avg, use_container_width=True)

            # Heatmap
            st.markdown("### Hourly Profile Heatmap (median kW)")
            hh["hour"] = hh["timestamp"].dt.hour
            pivot = (
                hh.groupby(["household_id", "hour"])["generation_kw"]
                .median()
                .unstack(fill_value=0)
            )
            fig_hm = px.imshow(
                pivot,
                aspect="auto",
                labels=dict(x="Hour of Day", y="Household", color="kW"),
                title="Median Output by Hour",
            )
            st.plotly_chart(fig_hm, use_container_width=True)
            st.caption("Rows with muted midday output hint at **shading**; all-day zeros can indicate **inverter failure**.")

# ===================== 5) Model Performance (headline) =====================
with tab5:
    st.subheader("Model Performance — Headline Metrics")

    # Pull from metrics.json if present; otherwise show abstract values
    mpath = "data/metrics.json"
    abstract_forecast_improvement = 13.27
    abstract_accuracy = 93.9

    if os.path.exists(mpath):
        try:
            with open(mpath) as f:
                M = json.load(f)
            f_mae_b = float(M["forecasting"]["baseline"]["mae"])
            f_mae_n = float(M["forecasting"]["neural"]["mae"])
            imp = M["forecasting"].get(
                "improvement_pct", 100.0 * (f_mae_b - f_mae_n) / max(f_mae_b, 1e-9)
            )
        except Exception:
            f_mae_b = np.nan
            f_mae_n = np.nan
            imp = abstract_forecast_improvement
    else:
        f_mae_b, f_mae_n, imp = np.nan, np.nan, abstract_forecast_improvement

    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline MAE", "—" if np.isnan(f_mae_b) else f"{f_mae_b:.3f}")
    c2.metric("Neural MAE", "—" if np.isnan(f_mae_n) else f"{f_mae_n:.3f}", f"−{imp:.2f}%")
    c3.write("Protocol: walk-forward validation across folds")

    if not np.isnan(f_mae_b) and not np.isnan(f_mae_n):
        df_perf = pd.DataFrame({"Model": ["Baseline", "Neural"], "MAE": [f_mae_b, f_mae_n]})
        fig_bar = px.bar(df_perf, x="Model", y="MAE", title="Forecasting MAE (lower is better)")
        st.plotly_chart(fig_bar, use_container_width=True)

    # Fault metrics (headline)
    fd_acc = abstract_accuracy
    fd_prec = None
    fd_rec = None
    if os.path.exists(mpath):
        try:
            with open(mpath) as f:
                M = json.load(f)
            fd_acc = float(M["fault_detection"]["accuracy"])
            fd_prec = float(M["fault_detection"].get("precision")) if "precision" in M["fault_detection"] else None
            fd_rec = float(M["fault_detection"].get("recall")) if "recall" in M["fault_detection"] else None
        except Exception:
            pass

    d1, d2, d3 = st.columns(3)
    d1.metric("Accuracy", f"{fd_acc:.1f}%")
    d2.metric("Precision", "—" if fd_prec is None else f"{fd_prec:.1f}%")
    d3.metric("Recall", "—" if fd_rec is None else f"{fd_rec:.1f}%")

    st.caption(
        "**Headline findings:** Neural forecasting reduced MAE by **13.27%** vs. linear baseline; "
        "fault detection achieved **93.9% accuracy**. Metrics computed offline; this app visualizes operator-facing outputs."
    )

# ===================== 6) Scheduling =====================
with tab6:
    st.subheader("Scheduling — Battery Dispatch (Toy Demo)")

    # Try to use the same solar forecast file if available
    fc_path = "data/forecast.csv"
    df_fc = load_csv(fc_path, parse_dates=["timestamp"])

    if df_fc is None or "generation_kw" not in (df_fc.columns if df_fc is not None else []):
        st.info(
            "Add **data/forecast.csv** with columns at least: "
            "`timestamp` (datetime) and `generation_kw` (float, kW)."
        )
    else:
        # Optional: load a simple demand profile if you have it, else assume flat demand
        # Supported columns for demand: timestamp, load_kw
        ld_path = "data/load.csv"
        df_ld = load_csv(ld_path, parse_dates=["timestamp"])
        has_load = df_ld is not None and "load_kw" in df_ld.columns

        if has_load:
            # align to forecast timestamps via left join
            df = pd.merge(df_fc[["timestamp", "generation_kw"]], df_ld[["timestamp", "load_kw"]], on="timestamp", how="left")
            # fill missing load with median or fallback
            if df["load_kw"].isna().any():
                df["load_kw"] = df["load_kw"].fillna(df["load_kw"].median() if df["load_kw"].notna().any() else 0.8)
        else:
            df = df_fc[["timestamp", "generation_kw"]].copy()
            df["load_kw"] = 0.8  # default flat demand for demo

        # Sort and basic hygiene
        df = df.sort_values("timestamp").reset_index(drop=True)

        # --- Controls ---
        st.markdown("#### Battery & Policy Settings")
        c1, c2, c3 = st.columns(3)
        with c1:
            cap_kwh = st.number_input("Battery capacity (kWh)", min_value=1.0, value=20.0, step=1.0)
            soc0 = st.number_input("Initial SoC (kWh)", min_value=0.0, value=10.0, step=0.5, help="State of charge at start")
        with c2:
            max_chg_kw = st.number_input("Max charge power (kW)", min_value=0.1, value=5.0, step=0.1)
            max_dis_kw = st.number_input("Max discharge power (kW)", min_value=0.1, value=5.0, step=0.1)
        with c3:
            eta_rt = st.slider("Round-trip efficiency (%)", min_value=50, max_value=100, value=92, step=1)
            target_soc_end = st.number_input("Target end SoC (kWh)", min_value=0.0, value=10.0, step=0.5)

        st.caption("Heuristic: charge on surplus solar, discharge to cover deficits; respects power limits and capacity.")

        # Infer time step (hours) from timestamps
        if len(df) >= 2:
            dt_seconds = (df["timestamp"].iloc[1] - df["timestamp"].iloc[0]).total_seconds()
            dt_h = max(dt_seconds / 3600.0, 1e-6)
        else:
            dt_h = 0.25  # fallback 15 min

        eta_c = np.sqrt((eta_rt / 100.0))   # charge eff
        eta_d = np.sqrt((eta_rt / 100.0))   # discharge eff

        # Prepare arrays
        soc = np.zeros(len(df), dtype=float)
        chg_kw = np.zeros(len(df), dtype=float)
        dis_kw = np.zeros(len(df), dtype=float)
        grid_kw = np.zeros(len(df), dtype=float)

        soc[0] = min(max(soc0, 0.0), cap_kwh)

        # Simple greedy policy
        for i in range(len(df)):
            gen = float(df.loc[i, "generation_kw"])
            load = float(df.loc[i, "load_kw"])
            net = gen - load  # + surplus, - deficit

            # try to keep SoC feasible and approach target by the end
            # For demo: if last 10% of horizon, bias SoC toward target
            bias = 0.0
            if len(df) >= 10 and i > 0.9 * (len(df) - 1):
                bias = (target_soc_end - soc[i-1]) / max((len(df) - i), 1) / dt_h if i > 0 else 0.0

            if net > 0:
                # charge up to surplus, limited by max power and headroom
                wanted = min(net + max(bias, 0.0), max_chg_kw)
                # energy we can actually store this step accounting for efficiency and headroom
                headroom_kwh = cap_kwh - (soc[i-1] if i > 0 else soc[0])
                max_store_kw = headroom_kwh / (dt_h * eta_c) if dt_h > 0 else 0.0
                p_chg = max(0.0, min(wanted, max_store_kw))
                chg_kw[i] = p_chg
                # update SoC
                delta = p_chg * dt_h * eta_c
                soc[i] = (soc[i-1] if i > 0 else soc[0]) + delta
                # any leftover surplus goes to grid export (negative grid import)
                grid_kw[i] = -(net - p_chg)
            else:
                # deficit: discharge to cover up to max power and available energy
                wanted = min(-net + max(-bias, 0.0), max_dis_kw)
                avail_kwh = (soc[i-1] if i > 0 else soc[0])
                max_draw_kw = avail_kwh * eta_d / dt_h if dt_h > 0 else 0.0
                p_dis = max(0.0, min(wanted, max_draw_kw))
                dis_kw[i] = p_dis
                delta = p_dis * dt_h / eta_d
                soc[i] = (soc[i-1] if i > 0 else soc[0]) - delta
                # any remaining deficit after discharge must be imported from grid (positive)
                grid_kw[i] = ( -net ) - p_dis

            # safety clamp
            soc[i] = min(max(soc[i], 0.0), cap_kwh)

            # carry SoC forward if there is a next step
            if i < len(df) - 1:
                df_idx_next = i + 1
                # prefill soc for next index so the next iteration has soc[i] as previous
                pass  # nothing needed; soc[i] already set

        schedule = pd.DataFrame({
            "timestamp": df["timestamp"],
            "generation_kw": df["generation_kw"],
            "load_kw": df["load_kw"],
            "charge_kw": chg_kw,
            "discharge_kw": dis_kw,
            "grid_import_kw": np.maximum(grid_kw, 0.0),
            "grid_export_kw": np.maximum(-grid_kw, 0.0),
            "soc_kwh": soc,
        })

        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Final SoC (kWh)", f"{schedule['soc_kwh'].iloc[-1]:.2f}")
        k2.metric("Total Grid Import (kWh)", f"{(schedule['grid_import_kw']*dt_h).sum():.2f}")
        k3.metric("Total Grid Export (kWh)", f"{(schedule['grid_export_kw']*dt_h).sum():.2f}")
        k4.metric("Round-trip η (%)", f"{eta_rt:.0f}%")

        # Plots
        fig_s1 = px.line(
            schedule, x="timestamp",
            y=["generation_kw", "load_kw", "charge_kw", "discharge_kw", "grid_import_kw", "grid_export_kw"],
            labels={"value": "kW", "variable": "Signal"},
            title="Dispatch Schedule (kW)"
        )
        st.plotly_chart(fig_s1, use_container_width=True)

        fig_s2 = px.line(
            schedule, x="timestamp", y="soc_kwh",
            labels={"soc_kwh": "kWh"},
            title="Battery State of Charge (kWh)"
        )
        st.plotly_chart(fig_s2, use_container_width=True)

        # Download
        csv_bytes = schedule.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download schedule CSV",
            data=csv_bytes,
            file_name="schedule.csv",
            mime="text/csv"
        )

        st.caption(
            "This is a **heuristic demo** (greedy charge/discharge with limits & efficiency). "
            "For research/ops you could replace it with MILP/LP optimization (e.g., Pyomo/Pulp) "
            "and real price signals."
        )

