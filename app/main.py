import base64
import streamlit as st
import sys
import os
from pathlib import Path
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml_package import data_processor, features, trainer, evaluation, plot_utils, exporter, models


# ── Helpers ───────────────────────────────────────────────────────────────────

def _img_b64(filename: str) -> str:
    path = Path(__file__).parent / filename
    return base64.b64encode(path.read_bytes()).decode() if path.exists() else ""


def _section_header(icon_b64: str, title: str, subtitle: str = "") -> None:
    sub = f'<p style="margin:0;color:#B3B3B3;font-size:0.9rem;">{subtitle}</p>' if subtitle else ""
    img_tag = (
        f'<img src="data:image/png;base64,{icon_b64}" width="64"'
        ' style="border-radius:8px;flex-shrink:0;">'
        if icon_b64 else ""
    )
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:18px;padding:4px 0 18px 0;">'
        f'{img_tag}'
        f'<div><h2 style="margin:0;color:#FFFFFF;">{title}</h2>{sub}</div>'
        f'</div>'
        f'<hr style="border-color:#2A2A2A;margin-bottom:1.2rem;">',
        unsafe_allow_html=True,
    )


def _white_bg(fig):
    """Apply white background with black, legible axis labels for EDA charts."""
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#111111", size=20),
        xaxis=dict(
            gridcolor="#e0e0e0",
            linecolor="#999999",
            tickfont=dict(color="#111111", size=20),
            title_font=dict(color="#111111", size=20),
        ),
        yaxis=dict(
            gridcolor="#e0e0e0",
            linecolor="#999999",
            tickfont=dict(color="#111111", size=20),
            title_font=dict(color="#111111", size=20),
        ),
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    logo_path = Path(__file__).parent / "icon_github.png"
    if logo_path.exists():
        st.sidebar.image(Image.open(logo_path), width=300)

    # ── About ─────────────────────────────────────────────────────────────────
    with st.sidebar.expander("👤 About", expanded=True):
        perfil_b64 = _img_b64("perfil.jpeg")
        photo_tag = (
            f'<img src="data:image/jpeg;base64,{perfil_b64}"'
            ' style="width:76px;height:76px;object-fit:cover;border-radius:50%;'
            'border:2px solid #1DB954;display:block;margin:0 auto 10px auto;">'
            if perfil_b64 else ""
        )

        pubs_sidebar = [
            (
                "2026",
                "Corn Plant Detection Using YOLOv9 Across Different Soil Backgrounds",
                "Remote Sensing",
                "https://doi.org/10.3390/rs18010014",
            ),
            (
                "2026",
                "Detection and Counting Marigold Flowers Using Drone Images and YOLOv8",
                "Ornamental Horticulture",
                "https://doi.org/10.1590/2447-536X.v32.e323014",
            ),
            (
                "2025",
                "Combining Flight Height and Nozzles to Enhance Arabica Coffee Spraying",
                "Proc. Indian National Science Academy",
                "https://doi.org/10.1007/s43538-025-00617-6",
            ),
            (
                "2025",
                "Impact of Atmospheric Corrections on Satellite Imagery for Corn Yield Prediction",
                "Smart Agricultural Technology",
                "https://doi.org/10.1016/j.atech.2025.101216",
            ),
        ]

        pub_rows_sb = "".join(
            f'<div style="padding:5px 0;border-top:1px solid #2A2A2A;">'
            f'<span style="color:#1DB954;font-size:0.68rem;font-weight:700;">{y}</span> '
            f'<a href="{doi}" target="_blank" style="color:#DDDDDD;text-decoration:none;'
            f'font-size:0.72rem;line-height:1.4;">{t}</a>'
            f'<br><span style="color:#777;font-size:0.65rem;font-style:italic;">{j}</span>'
            f'</div>'
            for y, t, j, doi in pubs_sidebar
        )

        st.markdown(
            f"""
            {photo_tag}
            <p style="text-align:center;margin:0 0 2px 0;color:#FFFFFF;
                      font-size:0.87rem;font-weight:600;">
              Thiago Orlando Costa Barboza
            </p>
            <p style="text-align:center;margin:0 0 10px 0;color:#1DB954;
                      font-size:0.75rem;line-height:1.5;">
              Ph.D. Candidate — Agronomy<br>
              <span style="color:#B3B3B3;font-size:0.7rem;">UFLA &amp; UGA &nbsp;·&nbsp;
              Brazil / USA</span>
            </p>

            <div style="display:flex;justify-content:space-around;
                        margin-bottom:12px;padding-bottom:10px;
                        border-bottom:1px solid #2A2A2A;">
              <div style="text-align:center;">
                <div style="color:#1DB954;font-weight:700;font-size:1.15rem;">16+</div>
                <div style="color:#888;font-size:0.62rem;text-transform:uppercase;
                            letter-spacing:0.05em;">Articles</div>
              </div>
              <div style="text-align:center;">
                <div style="color:#1DB954;font-weight:700;font-size:1.15rem;">22+</div>
                <div style="color:#888;font-size:0.62rem;text-transform:uppercase;
                            letter-spacing:0.05em;">Conf.</div>
              </div>
              <div style="text-align:center;">
                <div style="color:#1DB954;font-weight:700;font-size:1.15rem;">2</div>
                <div style="color:#888;font-size:0.62rem;text-transform:uppercase;
                            letter-spacing:0.05em;">Chapters</div>
              </div>
            </div>

            <p style="color:#888;font-size:0.63rem;font-weight:700;letter-spacing:0.1em;
                      text-transform:uppercase;margin:0 0 4px 0;">Recent Publications</p>
            {pub_rows_sb}

            <div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:10px;
                        padding-top:8px;border-top:1px solid #2A2A2A;">
              <a href="https://github.com/thiagoagro" target="_blank"
                 style="color:#1DB954;text-decoration:none;font-size:0.73rem;
                        font-weight:600;">🐙 GitHub</a>
              <a href="https://orcid.org/0000-0001-5156-2474" target="_blank"
                 style="color:#1DB954;text-decoration:none;font-size:0.73rem;
                        font-weight:600;">🔬 ORCID</a>
              <a href="http://lattes.cnpq.br/7295109791233637" target="_blank"
                 style="color:#1DB954;text-decoration:none;font-size:0.73rem;
                        font-weight:600;">📚 Lattes</a>
              <a href="http://www.linkedin.com/in/thiago-orlando-costa-barboza-5ba88213b"
                 target="_blank"
                 style="color:#1DB954;text-decoration:none;font-size:0.73rem;
                        font-weight:600;">💼 LinkedIn</a>
              <a href="mailto:thiagocostaagro@gmail.com"
                 style="color:#1DB954;text-decoration:none;font-size:0.73rem;
                        font-weight:600;">✉️ Email</a>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Researcher header (above page title) ─────────────────────────────────────

def _render_researcher_header() -> None:
    perfil_b64 = _img_b64("perfil.jpeg")
    photo_tag = (
        f'<img src="data:image/jpeg;base64,{perfil_b64}"'
        ' style="width:110px;height:110px;object-fit:cover;border-radius:50%;'
        'border:2px solid #1DB954;flex-shrink:0;">'
        if perfil_b64 else ""
    )

    pubs = [
        ("2026",
         "Corn Plant Detection Using YOLOv9 Across Different Soil Backgrounds, Growth Stages and UAV Heights",
         "Remote Sensing",
         "https://doi.org/10.3390/rs18010014"),
        ("2026",
         "Detection and Counting Marigold Flowers Using Drone Images and YOLOv8 in Complex Environments",
         "Ornamental Horticulture",
         "https://doi.org/10.1590/2447-536X.v32.e323014"),
        ("2025",
         "Combining Flight Height and Nozzles to Enhance Arabica Coffee Spraying Efficiency with Drones",
         "Proc. Indian National Science Academy",
         "https://doi.org/10.1007/s43538-025-00617-6"),
        ("2025",
         "Impact of Atmospheric Corrections on Satellite Imagery for Corn Yield Prediction via ML",
         "Smart Agricultural Technology",
         "https://doi.org/10.1016/j.atech.2025.101216"),
        ("2025",
         "Application of Artificial Intelligence for Identification of Peanut Maturity Using Climatic Variables",
         "Precision Agriculture",
         "https://doi.org/10.1007/s11119-025-10237-1"),
    ]

    pub_rows = "".join(
        f'<div style="display:flex;align-items:baseline;gap:10px;'
        f'padding:7px 0;border-top:1px solid #2A2A2A;">'
        f'<span style="color:#1DB954;font-size:0.72rem;font-weight:700;'
        f'white-space:nowrap;min-width:32px;">{y}</span>'
        f'<div><a href="{doi}" target="_blank" style="color:#FFFFFF;text-decoration:none;'
        f'font-size:0.82rem;line-height:1.4;">{t}</a>'
        f'<br><span style="color:#777;font-size:0.72rem;font-style:italic;">{j}</span></div>'
        f'</div>'
        for y, t, j, doi in pubs
    )

    st.markdown(
        f"""
        <div style="background:#181818;border:1px solid #2A2A2A;border-radius:1rem;
                    padding:1.6rem 2rem;margin-bottom:1.5rem;">

          <!-- Row 1: photo + name + metrics -->
          <div style="display:flex;align-items:center;gap:1.8rem;flex-wrap:wrap;
                      margin-bottom:1.4rem;">
            {photo_tag}
            <div style="flex:1;min-width:180px;">
              <h3 style="margin:0 0 2px 0;color:#FFFFFF;font-size:1.15rem;">
                Thiago Orlando Costa Barboza
              </h3>
              <p style="margin:0 0 2px 0;color:#1DB954;font-size:0.82rem;font-weight:600;">
                Ph.D. Candidate in Agronomy
              </p>
              <p style="margin:0;color:#B3B3B3;font-size:0.76rem;">
                Precision &amp; Digital Agriculture &nbsp;·&nbsp; UFLA &amp; UGA &nbsp;·&nbsp; Brazil / USA
              </p>
            </div>
            <div style="display:flex;gap:1.5rem;flex-wrap:wrap;">
              <div style="text-align:center;">
                <div style="color:#1DB954;font-size:1.5rem;font-weight:700;">16+</div>
                <div style="color:#B3B3B3;font-size:0.68rem;text-transform:uppercase;
                            letter-spacing:0.06em;">Journal<br>Articles</div>
              </div>
              <div style="text-align:center;">
                <div style="color:#1DB954;font-size:1.5rem;font-weight:700;">22+</div>
                <div style="color:#B3B3B3;font-size:0.68rem;text-transform:uppercase;
                            letter-spacing:0.06em;">Conference<br>Presentations</div>
              </div>
              <div style="text-align:center;">
                <div style="color:#1DB954;font-size:1.5rem;font-weight:700;">2</div>
                <div style="color:#B3B3B3;font-size:0.68rem;text-transform:uppercase;
                            letter-spacing:0.06em;">Book<br>Chapters</div>
              </div>
            </div>
          </div>

          <!-- Row 2: Recent publications -->
          <div style="margin-bottom:1.2rem;">
            <p style="color:#B3B3B3;font-size:0.7rem;font-weight:700;letter-spacing:0.1em;
                      text-transform:uppercase;margin:0 0 4px 0;">Recent Publications</p>
            {pub_rows}
          </div>

          <!-- Row 3: Links -->
          <div style="display:flex;gap:1.2rem;flex-wrap:wrap;padding-top:10px;
                      border-top:1px solid #2A2A2A;">
            <a href="https://github.com/thiagoagro" target="_blank"
               style="color:#1DB954;text-decoration:none;font-size:0.8rem;font-weight:600;">
              🐙 GitHub
            </a>
            <a href="https://orcid.org/0000-0001-5156-2474" target="_blank"
               style="color:#1DB954;text-decoration:none;font-size:0.8rem;font-weight:600;">
              🔬 ORCID
            </a>
            <a href="http://lattes.cnpq.br/7295109791233637" target="_blank"
               style="color:#1DB954;text-decoration:none;font-size:0.8rem;font-weight:600;">
              📚 Lattes
            </a>
            <a href="http://www.linkedin.com/in/thiago-orlando-costa-barboza-5ba88213b"
               target="_blank"
               style="color:#1DB954;text-decoration:none;font-size:0.8rem;font-weight:600;">
              💼 LinkedIn
            </a>
            <a href="mailto:thiagocostaagro@gmail.com"
               style="color:#1DB954;text-decoration:none;font-size:0.8rem;font-weight:600;">
              ✉️ Email
            </a>
          </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Platform identity hero ────────────────────────────────────────────────────

def _render_platform_identity() -> None:
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#061306 0%,#181818 55%,#061306 100%);
                    border:1px solid #1DB954;border-radius:1rem;
                    padding:2rem 2.5rem;margin:0.5rem 0 1.8rem 0;">
          <div style="color:#1DB954;font-size:0.75rem;font-weight:700;
                      letter-spacing:0.18em;text-transform:uppercase;margin-bottom:0.6rem;">
            Open Source &nbsp;·&nbsp; Precision Agriculture &nbsp;·&nbsp; No-Code ML
          </div>
          <h2 style="margin:0 0 0.8rem 0;color:#FFFFFF;font-size:1.55rem;line-height:1.35;">
            From raw data to trained models.<br>
            <span style="color:#1DB954;">No code required.</span>
          </h2>
          <p style="margin:0 0 1.2rem 0;color:#B3B3B3;font-size:0.88rem;line-height:1.7;
                    max-width:700px;">
            <strong style="color:#FFFFFF;">ML Package Platform (MPP)</strong> was built by an
            agricultural researcher frustrated with the repetitive overhead of training and
            comparing ML models — adapting scikit-learn, XGBoost, LightGBM and CatBoost across
            dozens of experiments for yield prediction, crop monitoring, and UAV-based phenotyping.
            MPP eliminates that friction: upload your dataset, select your target variable, and let
            the platform handle EDA, feature selection, hyperparameter tuning and evaluation — all
            within a single interface, no code written.
          </p>
          <div style="display:flex;gap:2rem;flex-wrap:wrap;align-items:flex-start;">
            <div style="text-align:center;">
              <div style="color:#1DB954;font-size:1.4rem;font-weight:700;">8+</div>
              <div style="color:#B3B3B3;font-size:0.7rem;">ML Algorithms</div>
            </div>
            <div style="text-align:center;">
              <div style="color:#1DB954;font-size:1.4rem;font-weight:700;">2</div>
              <div style="color:#B3B3B3;font-size:0.7rem;">Task Types</div>
            </div>
            <div style="text-align:center;">
              <div style="color:#1DB954;font-size:1.4rem;font-weight:700;">AutoML</div>
              <div style="color:#B3B3B3;font-size:0.7rem;">Hyperparameter Tuning</div>
            </div>
            <div style="text-align:center;">
              <div style="color:#1DB954;font-size:1.4rem;font-weight:700;">Excel</div>
              <div style="color:#B3B3B3;font-size:0.7rem;">One-Click Export</div>
            </div>
            <div style="text-align:center;">
              <div style="color:#1DB954;font-size:1.4rem;font-weight:700;">100%</div>
              <div style="color:#B3B3B3;font-size:0.7rem;">Open Source</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="ML Package Platform",
        page_icon="🤖",
        layout="wide",
    )

    # ── CSS: tech-card expanders ──────────────────────────────────────────────
    st.markdown(
        """<style>
        [data-testid="stExpander"] {
            border: 1px solid #1DB954 !important;
            border-radius: 1rem !important;
            margin-bottom: 1.2rem !important;
            background: #181818 !important;
            overflow: hidden;
        }
        [data-testid="stExpander"] details summary {
            font-size: 1.25rem !important;
            padding: 1.2rem 1.8rem !important;
            color: #1DB954 !important;
            font-weight: 600 !important;
        }
        [data-testid="stExpander"] details summary:hover {
            background: rgba(29,185,84,0.08) !important;
        }
        [data-testid="stExpander"] details[open] summary {
            border-bottom: 1px solid #2A2A2A;
        }
        </style>""",
        unsafe_allow_html=True,
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    _render_sidebar()

    st.sidebar.markdown("---")
    st.sidebar.markdown("##### 📂 Dataset")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your dataset", type=["csv", "xlsx", "xls"]
    )

    # ── Main content ──────────────────────────────────────────────────────────
    _render_researcher_header()
    st.title("Machine Learning Platform")
    _render_platform_identity()

    with st.expander("ℹ️ How to use this platform"):
        st.markdown(
            """
            1. **Upload** your dataset (`.csv` or `.xlsx`) in the sidebar.
            2. **EDA** — Explore distributions and variable correlations.
            3. **Feature Selection** — Identify the most impactful variables for your target.
            4. **Training Setup** — Choose features, scaling method, models, and hit *Start Training*.
            5. **Results** — View the model leaderboard, interactive charts, and download reports.
            """
        )

    if not uploaded_file:
        st.info("Please upload a dataset from the sidebar to get started.")
        return

    df = data_processor.load_data(uploaded_file)

    # ── NaN handling ─────────────────────────────────────────────────────────
    _file_key = f"_nan_clean_{uploaded_file.name}_{uploaded_file.size}"
    nan_counts = df.isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]

    if len(nan_cols) > 0:
        if _file_key in st.session_state:
            df = st.session_state[_file_key]
        else:
            st.warning(
                f"**Missing values detected** in {len(nan_cols)} column(s). "
                "Choose how to handle them before proceeding."
            )
            c_tbl, c_act = st.columns([1, 1])
            with c_tbl:
                st.markdown("**Columns with missing values:**")
                st.dataframe(
                    nan_cols.rename("Missing count").to_frame(),
                    use_container_width=True,
                )
            with c_act:
                nan_strategy = st.radio(
                    "Strategy",
                    [
                        "Replace with column median",
                        "Drop rows with any missing value",
                    ],
                    key="nan_strategy",
                )
                if st.button("Apply and continue ▶", type="primary", key="apply_nan"):
                    if nan_strategy == "Replace with column median":
                        df = df.fillna(df.median(numeric_only=True))
                    else:
                        df = df.dropna()
                    st.session_state[_file_key] = df
                    st.rerun()
                else:
                    st.stop()
    else:
        st.session_state.pop(_file_key, None)

    st.sidebar.success(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    eda_b64   = _img_b64("eda_icon.png")
    fs_b64    = _img_b64("fs_icon.png")
    train_b64 = _img_b64("train_icon.png")
    res_b64   = _img_b64("results_icon.png")

    # ── SECTION 1: EDA ────────────────────────────────────────────────────────
    with st.expander("📊  EDA — Exploratory Data Analysis", expanded=True):
        _section_header(
            eda_b64,
            "Exploratory Data Analysis",
            "Explore distributions, correlations, and data quality",
        )
        eda_info = data_processor.generate_eda(df)

        col_preview, col_missing = st.columns(2)
        with col_preview:
            st.subheader("Data Preview")
            st.dataframe(eda_info["head"])
        with col_missing:
            st.subheader("Missing Values")
            if len(eda_info["missing_values"]) > 0:
                st.dataframe(eda_info["missing_values"].rename("Count"))
            else:
                st.success("No missing values found in this dataset.")

        st.subheader("Interactive Charts")
        num_cols = eda_info["numeric_cols"]

        if len(num_cols) > 1:
            col_x_sel, col_y_sel = st.columns(2)
            with col_x_sel:
                x_axis = st.selectbox("X Axis (Scatter / Histogram)", num_cols, index=0)
            with col_y_sel:
                y_axis = st.selectbox(
                    "Y Axis (Scatter only)", num_cols, index=min(1, len(num_cols) - 1)
                )

            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.plotly_chart(
                    _white_bg(plot_utils.plot_interactive_scatter(df, x_axis, y_axis)),
                    use_container_width=True,
                )
            with col_chart2:
                st.plotly_chart(
                    _white_bg(plot_utils.plot_interactive_histogram(df, x_axis)),
                    use_container_width=True,
                )

            st.subheader("Global Correlation Matrix")
            st.plotly_chart(
                _white_bg(plot_utils.plot_correlation_matrix(df)),
                use_container_width=True,
            )
        else:
            st.warning("Not enough numeric columns to display charts.")

    # ── SECTION 2: Feature Selection ──────────────────────────────────────────
    with st.expander("🎯  Feature Selection", expanded=False):
        _section_header(
            fs_b64,
            "Intelligent Feature Selection",
            "Compare multiple selection methods and identify the most impactful variables",
        )

        # ── Config row ────────────────────────────────────────────────────────
        cfg_col1, cfg_col2 = st.columns([2, 1])
        with cfg_col1:
            fs_target = st.selectbox(
                "Target variable (what you want to predict)",
                df.columns, key="fs_target",
            )
        with cfg_col2:
            fs_task = st.radio(
                "Task type", ["Regression", "Classification"], key="fs_task"
            )

        # ── Method checkboxes ─────────────────────────────────────────────────
        st.markdown("**Select feature selection methods to run:**")
        method_labels = list(features.FEATURE_SELECTION_METHODS.keys())
        selected_methods = st.multiselect(
            "Methods",
            options=method_labels,
            default=["ExtraTrees Importance", "Mutual Information"],
            help=(
                "ExtraTrees / Random Forest: tree-based importance.  "
                "Mutual Information: non-linear statistical dependency.  "
                "F-test: linear ANOVA/F-regression.  "
                "RFE: iterative feature elimination.  "
                "Lasso / L1: regularization-based shrinkage."
            ),
            label_visibility="collapsed",
        )

        if st.button("▶  Run Feature Selection", type="primary"):
            if not selected_methods:
                st.warning("Select at least one method.")
            else:
                X_temp = df.drop(columns=[fs_target])
                X_num  = X_temp.select_dtypes(include=["number"])
                y_temp = df[fs_target]

                if X_num.shape[1] == 0:
                    st.warning("No numeric feature columns available.")
                else:
                    fs_results = {}
                    progress = st.progress(0, text="Running feature selection…")
                    for i, label in enumerate(selected_methods):
                        key = features.FEATURE_SELECTION_METHODS[label]
                        progress.progress(
                            int((i / len(selected_methods)) * 100),
                            text=f"Running {label}…",
                        )
                        try:
                            fs_results[label] = features.run_feature_selection(
                                X_num, y_temp, fs_task, key
                            )
                        except Exception as exc:
                            st.warning(f"**{label}** failed: {exc}")
                    progress.empty()

                    st.session_state["fs_results"] = fs_results

                    # Suggestion from first method (usually ExtraTrees)
                    first_df = next(iter(fs_results.values()))
                    top_features = first_df[
                        first_df["Cumulative_Importance"] <= 0.95
                    ]["Feature"].tolist()
                    if not top_features:
                        top_features = first_df["Feature"].head(5).tolist()
                    st.session_state["suggested_features"] = top_features

        # ── Display results ───────────────────────────────────────────────────
        fs_results = st.session_state.get("fs_results", {})
        if fs_results:
            # Individual charts
            if len(fs_results) == 1:
                label, res_df_fs = next(iter(fs_results.items()))
                st.plotly_chart(
                    plot_utils.plot_fs_single(res_df_fs, label),
                    use_container_width=True,
                )
            else:
                tabs_fs = st.tabs(list(fs_results.keys()))
                for tab, (label, res_df_fs) in zip(tabs_fs, fs_results.items()):
                    with tab:
                        st.plotly_chart(
                            plot_utils.plot_fs_single(res_df_fs, label),
                            use_container_width=True,
                        )

                # Comparison chart
                st.markdown("#### Comparison — All Methods (Normalized Scores)")
                st.plotly_chart(
                    plot_utils.plot_fs_comparison(fs_results),
                    use_container_width=True,
                )

            # Suggestion banner
            top_features = st.session_state.get("suggested_features", [])
            if top_features:
                st.info(
                    f"**Suggestion (based on {next(iter(fs_results))}):** "
                    f"Variables explaining 95 % of predictive information: "
                    f"**{', '.join(top_features)}**"
                )

    # ── SECTION 3: Training Setup ─────────────────────────────────────────────
    with st.expander("⚙️  Training Setup", expanded=False):
        _section_header(
            train_b64,
            "Training Configuration",
            "Select features, models, and hyperparameter strategy",
        )

        task_type = st.radio(
            "Task type",
            ["Regression", "Classification"],
            index=0 if st.session_state.get("fs_task", "Regression") == "Regression" else 1,
        )

        available_models = (
            list(models.get_regression_models().keys())
            if task_type == "Regression"
            else list(models.get_classification_models().keys())
        )

        # ── HPO config (outside form — editable anytime) ─────────────────────
        hpo_choice = st.radio(
            "Hyperparameter optimisation strategy",
            ["Disabled", "Random Search", "Optuna — TPE (Bayesian)"],
            index=1,
            horizontal=True,
            help=(
                "**Random Search:** samples param combinations randomly — fast baseline.  \n"
                "**Optuna (TPE):** Bayesian optimisation that learns from previous trials — "
                "finds better hyperparameters in fewer evaluations."
            ),
        )
        n_trials = st.slider(
            "Optuna trials per model",
            min_value=10, max_value=100, value=30, step=5,
            disabled="Optuna" not in hpo_choice,
            help="More trials → better results but longer runtime. 30–50 is a good range.",
        )

        with st.form("training_form"):
            target_col = st.selectbox(
                "Target variable",
                df.columns,
                index=int(
                    df.columns.get_loc(
                        st.session_state.get("fs_target", df.columns[0])
                    )
                ),
            )

            default_feats = st.session_state.get(
                "suggested_features",
                [c for c in df.columns if c != target_col],
            )
            default_feats = [
                f for f in default_feats if f in df.columns and f != target_col
            ]

            features_cols = st.multiselect(
                "Input features",
                [c for c in df.columns if c != target_col],
                default=default_feats,
            )
            scaler_option = st.selectbox(
                "Scaling method",
                ["StandardScaler", "MinMaxScaler", "RobustScaler", "None"],
            )
            selected_models_list = st.multiselect(
                "Models to train",
                available_models,
                default=available_models,
                help="Deselect models you do not want to include.",
            )
            submit_btn = st.form_submit_button("Start Training 🚀")

        if submit_btn:
            if not features_cols:
                st.error("Please select at least one input feature.")
            elif not selected_models_list:
                st.error("Please select at least one model.")
            else:
                _hpo_map = {
                    "Disabled": "none",
                    "Random Search": "random_search",
                    "Optuna — TPE (Bayesian)": "optuna",
                }
                _spinner_msg = (
                    f"Running Optuna ({n_trials} trials/model) — this may take a few minutes…"
                    if "Optuna" in hpo_choice
                    else "Training models… this may take a few minutes."
                )
                with st.spinner(_spinner_msg):
                    results_df, preds_df, trained_models_dict, optuna_studies = (
                        trainer.run_training_pipeline(
                            df,
                            features_cols,
                            target_col,
                            task_type,
                            scaler_option,
                            hpo_method=_hpo_map[hpo_choice],
                            n_trials=n_trials,
                            selected_models=selected_models_list,
                        )
                    )
                    st.session_state["results_df"] = results_df
                    st.session_state["preds_df"] = preds_df
                    st.session_state["task_type"] = task_type
                    st.session_state["trained_models"] = trained_models_dict
                    st.session_state["optuna_studies"] = optuna_studies
                    st.session_state["training_done"] = True
                st.success("Training complete! Open the **Results** section below.")

    # ── SECTION 4: Results ────────────────────────────────────────────────────
    with st.expander("🚀  Results & Export", expanded=False):
        _section_header(
            res_b64,
            "Results & Export",
            "Model leaderboard, performance charts, and downloads",
        )

        if not st.session_state.get("training_done"):
            st.info(
                "No training has been run yet. "
                "Configure your pipeline in the Training Setup section."
            )
        else:
            res_df = st.session_state["results_df"]
            preds = st.session_state["preds_df"]
            trained_models_dict = st.session_state.get("trained_models", {})

            st.subheader("Model Leaderboard (Test Metrics)")
            st.dataframe(res_df.style.background_gradient(cmap="Greens"))

            st.subheader("Visual Performance — Top 5 Models")
            top_5 = res_df["Model"].head(5).tolist()

            for m_name in top_5:
                st.markdown(f"#### {m_name}")
                m_preds = preds[preds["Model"] == m_name]

                if st.session_state["task_type"] == "Regression":
                    fig = _white_bg(plot_utils.plot_regression_performance(m_preds, m_name))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    m_test = m_preds[m_preds["Dataset"] == "Test"]
                    labels = sorted(m_test["Observed"].unique())
                    fig = _white_bg(plot_utils.plot_confusion_matrix_plotly(
                        m_test["Observed"],
                        m_test["Predicted"],
                        labels,
                        f"Confusion Matrix: {m_name}",
                    ))
                    st.plotly_chart(fig, use_container_width=True)

            # ── Metrics summary chart ──────────────────────────────────────
            st.subheader("Metrics Summary — All Models")
            summary_fig = plot_utils.plot_metrics_summary(
                res_df, st.session_state["task_type"]
            )
            if summary_fig is not None:
                st.plotly_chart(summary_fig, use_container_width=True)

            # ── Optuna HPO details ─────────────────────────────────────────
            optuna_studies = st.session_state.get("optuna_studies")
            if optuna_studies:
                st.subheader("Hyperparameter Optimisation Details (Optuna — TPE)")
                for m_name, study in optuna_studies.items():
                    with st.expander(
                        f"📈 {m_name} — {len(study.trials)} trials · "
                        f"best CV = {study.best_value:.4f}"
                    ):
                        col_hist, col_params = st.columns([2, 1])
                        with col_hist:
                            try:
                                import optuna
                                hist_fig = optuna.visualization.plot_optimization_history(study)
                                hist_fig = _white_bg(hist_fig)
                                st.plotly_chart(hist_fig, use_container_width=True)
                            except Exception:
                                st.info("Optimization history chart unavailable.")
                        with col_params:
                            st.markdown("**Best Parameters Found:**")
                            for k, v in study.best_params.items():
                                st.markdown(f"- `{k}`: **{v}**")
                            st.metric("Best CV Score", f"{study.best_value:.4f}")

            st.subheader("Download Reports")
            excel_bytes = exporter.export_to_excel(res_df, preds, None)
            st.download_button(
                label="📥 Download Excel Report (.xlsx)",
                data=excel_bytes,
                file_name="ML_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            if trained_models_dict:
                st.markdown("---")
                st.subheader("Save Trained Models (.joblib)")
                st.write(
                    "Download individual fitted models to reuse them later "
                    "(`joblib.load('model.joblib')`)."
                )
                model_cols = st.columns(min(3, len(trained_models_dict)))
                for idx, (m_name, fitted_model) in enumerate(trained_models_dict.items()):
                    safe_name = m_name.replace(" ", "_").replace("(", "").replace(")", "")
                    model_bytes = trainer.model_to_bytes(fitted_model)
                    with model_cols[idx % len(model_cols)]:
                        st.download_button(
                            label=f"💾 {m_name}",
                            data=model_bytes,
                            file_name=f"{safe_name}.joblib",
                            mime="application/octet-stream",
                            key=f"dl_model_{safe_name}",
                        )


if __name__ == "__main__":
    main()
