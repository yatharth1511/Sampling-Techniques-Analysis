import streamlit as st
import requests
import pandas as pd
import os
import zipfile
from plots import plot_model_wise_accuracy, plot_accuracy_heatmap

st.set_page_config(
    page_title="Sampling Analysis Dashboard",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

div[data-testid="stVerticalBlock"]:empty {
    display: none !important;
}

div[data-testid="stExpander"] div[data-testid="stVerticalBlock"]:empty {
    display: none !important;
}

h1 + div[data-testid="stVerticalBlock"],
h2 + div[data-testid="stVerticalBlock"],
h3 + div[data-testid="stVerticalBlock"] {
    display: none !important;
}

.card {
    background-color: white;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    margin-bottom: 1.5rem;
}

.title {
    text-align: center;
    font-size: 2.2rem;
    font-weight: 700;
}

.subtitle {
    text-align: center;
    color: #9ca3af;
    margin-bottom: 2rem;
}

</style>
""", unsafe_allow_html=True)


st.markdown("<div class='title'>📊 Sampling Techniques Analysis</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Interactive Dashboard for Model Accuracy Comparison</div>", unsafe_allow_html=True)

st.sidebar.header("⚙ Controls")

run_exp = st.sidebar.button("🚀 Run Experiment")

show_table = st.sidebar.checkbox("Show Accuracy Table", True)
show_plots = st.sidebar.checkbox("Show Plots", True)

if run_exp:
    with st.spinner("Running experiments..."):
        response = requests.get("http://localhost:8000/run")
        results = response.json()

    df = pd.DataFrame(results).T
    st.session_state["results_df"] = df

if "results_df" in st.session_state:
    df = st.session_state["results_df"]

    st.sidebar.subheader("🎯 Filters")
    selected_models = st.sidebar.multiselect(
        "Select Models",
        options=list(df.index),
        default=list(df.index)
    )

    selected_sampling = st.sidebar.multiselect(
        "Select Sampling Techniques",
        options=list(df.columns),
        default=list(df.columns)
    )

    filtered_df = df.loc[selected_models, selected_sampling]

    if show_table:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📋 Accuracy Results")
        st.dataframe(filtered_df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        csv = filtered_df.to_csv().encode("utf-8")
        st.download_button(
            "📥 Download Results as CSV",
            csv,
            file_name="sampling_accuracy_results.csv",
            mime="text/csv"
        )

    if show_plots:
        with st.expander("📈 Visual Analysis"):
            os.makedirs("exports", exist_ok=True)
            png_files = []

            for fig in plot_model_wise_accuracy(filtered_df):
                st.pyplot(fig)
                fname = f"exports/model_plot_{len(png_files)}.png"
                fig.savefig(fname)
                png_files.append(fname)

            heatmap_fig = plot_accuracy_heatmap(filtered_df)
            st.pyplot(heatmap_fig)
            heatmap_path = "exports/accuracy_heatmap.png"
            heatmap_fig.savefig(heatmap_path)
            png_files.append(heatmap_path)

            zip_path = "exports/plots.zip"
            with zipfile.ZipFile(zip_path, "w") as zipf:
                for file in png_files:
                    zipf.write(file, arcname=os.path.basename(file))

            with open(zip_path, "rb") as f:
                st.download_button(
                    "🖼 Download Plots as PNG (ZIP)",
                    f,
                    file_name="accuracy_plots.zip",
                    mime="application/zip"
                )

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🏆 Best Sampling Technique per Model")
    best_sampling = filtered_df.idxmax(axis=1)
    st.write(best_sampling)
    st.markdown("</div>", unsafe_allow_html=True)
