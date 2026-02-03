import matplotlib.pyplot as plt
import pandas as pd

def plot_model_wise_accuracy(df):
    figures = []

    for model in df.index:
        fig, ax = plt.subplots()
        df.loc[model].plot(kind="bar", ax=ax)
        ax.set_title(f"Accuracy Comparison for {model}")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlabel("Sampling Techniques")
        ax.set_ylim(0, 100)
        figures.append(fig)

    return figures


def plot_accuracy_heatmap(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(df.values)

    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)

    ax.set_title("Sampling Techniques vs ML Models")
    fig.colorbar(im, ax=ax, label="Accuracy (%)")

    return fig
