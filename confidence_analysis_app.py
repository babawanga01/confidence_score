import streamlit as st
import pandas as pd
import numpy as np

# --- Compact UI tweaks ---
st.markdown(
    """
    <style>
    div[data-testid='stSlider'] {margin-top: 0.2rem; margin-bottom: 0.2rem;}
    div[data-testid='stSlider'] label {margin-bottom: 0.1rem; font-size: 0.9rem;}
    .block-container h3 {margin-top: 0.25rem; margin-bottom: 0.25rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Confidence Score Analysis - Interactive Bins")

uploaded_file = st.file_uploader(
    "Choose an Excel file",
    type="xlsx",
    help="Upload your Excel file containing confidence_score and Rating columns",
)

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, sheet_name="Sheet1")
        df_clean = df[["confidence_score", "Rating"]].dropna()
        df_clean = df_clean[pd.to_numeric(df_clean["Rating"], errors="coerce").notna()]
        df_clean["Rating"] = df_clean["Rating"].astype(int)
        st.success(f"Data loaded successfully! Total records: {len(df_clean)}")

        st.sidebar.header("Bin Configuration")
        bin_method = st.sidebar.selectbox(
            "Select Binning Method:",
            ["Custom Bins (Sliders)", "Equal Width", "Equal Frequency", "Quantiles"],
        )

        bins = None
        if bin_method == "Custom Bins (Sliders)":
            st.sidebar.subheader("Custom Bin Edges")
            st.sidebar.caption(
                "Drag sliders to set internal edges between 0.00 and 1.00."
            )
            num_bins = st.sidebar.slider(
                "Number of bins", min_value=2, max_value=10, value=5
            )
            step = 0.05

            # Default inner edges
            default_edges = np.linspace(0, 1, num_bins + 1)[1:-1]

            st.markdown("### Custom edges")
            st.markdown(
                "Move each edge; spacing is enforced at 0.05. Values show 2 decimals."
            )

            inner_count = num_bins - 1
            cols = st.columns(inner_count, vertical_alignment="bottom")

            # Render sliders left-to-right and collect values; compute bounds using previously chosen value
            proposed = []
            left_anchor = 0.0
            for i in range(inner_count):
                with cols[i]:
                    left_bound = round(left_anchor + step, 2)
                    right_bound = round(1.0 - step * (inner_count - i), 2)
                    if left_bound > right_bound:
                        left_bound = max(0.0, right_bound - step)
                    default_val = float(
                        np.clip(
                            round(float(default_edges[i]), 2), left_bound, right_bound
                        )
                    )
                    val = st.slider(
                        f"Edge {i + 1}",
                        min_value=left_bound,
                        max_value=right_bound,
                        value=default_val,
                        step=step,
                        format="%.2f",
                        key=f"edge_{i + 1}",
                    )
                    proposed.append(val)
                    left_anchor = val

            inner_sorted = sorted([round(v, 2) for v in proposed])
            bins = [0.0] + inner_sorted + [1.0]

        elif bin_method == "Equal Width":
            num_bins = st.sidebar.slider(
                "Number of bins", min_value=2, max_value=10, value=5
            )
            bins = np.linspace(0, 1, num_bins + 1).round(2).tolist()

        elif bin_method == "Equal Frequency":
            num_bins = st.sidebar.slider(
                "Number of bins", min_value=2, max_value=10, value=5
            )
            inner = pd.qcut(
                df_clean["confidence_score"],
                q=num_bins - 1,
                retbins=True,
                duplicates="drop",
            )[1][1:-1]
            bins = [0.0] + [round(float(x), 2) for x in inner] + [1.0]

        elif bin_method == "Quantiles":
            percentiles = st.sidebar.multiselect(
                "Select percentiles (0 and 100 included)",
                [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90],
                default=[25, 50, 75],
            )
            percentiles = sorted([0] + percentiles + [100])
            vals = [np.percentile(df_clean["confidence_score"], p) for p in percentiles]
            bins = [round(float(x), 2) for x in vals]

        st.sidebar.write("Current bins:", bins)

        # Labels
        labels = [f"{bins[i]:.2f}-{bins[i + 1]:.2f}" for i in range(len(bins) - 1)]

        # Bin, then remove unused categories to avoid empty rows
        binned = pd.cut(
            df_clean["confidence_score"],
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=False,
        )
        binned = binned.cat.remove_unused_categories()
        df_clean["confidence_score_Bin"] = binned

        # Crosstab
        crosstab = pd.crosstab(
            df_clean["confidence_score_Bin"], df_clean["Rating"], margins=True
        )

        # Accuracy
        def calc_accuracy(row):
            weights = {0: 0, 1: 20, 2: 40, 3: 60, 4: 80, 5: 100}
            total = 0
            count = 0
            for rating, weight in weights.items():
                n = row.get(rating, 0)
                total += n * weight
                count += n
            return round(total / count, 1) if count > 0 else None

        crosstab["Accuracy %"] = crosstab.apply(calc_accuracy, axis=1)
        grand_total = crosstab.loc["All", "All"]
        crosstab["% Distribution"] = (
            ((crosstab["All"] / grand_total) * 100).round(0).astype(int)
        )

        st.header("Cross-tabulation Results")
        st.dataframe(
            crosstab.style.format({"Accuracy %": "{:.1f}", "% Distribution": "{:.0f}"}),
            use_container_width=True,
            height=420,
        )

        st.header("Key Insights")
        col1, col2, col3 = st.columns(3)
        with col1:
            best_bin = crosstab["Accuracy %"][:-1].idxmax()
            st.metric(
                "Best Performing Bin",
                str(best_bin),
                f"{crosstab.loc[best_bin, 'Accuracy %']}%",
            )
        with col2:
            largest_bin = crosstab["% Distribution"][:-1].idxmax()
            st.metric(
                "Largest Bin",
                str(largest_bin),
                f"{crosstab.loc[largest_bin, '% Distribution']}%",
            )
        with col3:
            st.metric("Overall Accuracy", f"{crosstab.loc['All', 'Accuracy %']}%")

        st.header("Download Results")
        st.download_button(
            label="Download CSV",
            data=crosstab.to_csv(),
            file_name="confidence_score_analysis.csv",
            mime="text/csv",
        )

        with st.expander("View Raw Data Preview"):
            st.dataframe(df_clean.head(100))

    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        st.info(
            "Please ensure your Excel file has 'confidence_score' and 'Rating' columns in Sheet1"
        )
else:
    st.info("Please upload an Excel file to begin analysis")
    st.subheader("Expected Data Format")
    st.dataframe(
        pd.DataFrame(
            {
                "confidence_score": [0.45, 0.67, 0.82, 0.34, 0.91],
                "Rating": [2, 3, 4, 1, 5],
            }
        )
    )
