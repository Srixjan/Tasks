import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mini_project_mplads_fund_eda.main import build_final_dataframe


def plot_utlization_rate(df: pd.DataFrame) -> None:
    plt.style.use("bmh")
    plt.hist(df["utilization_rate"], bins=15, color="royalblue", edgecolor="black")
    plt.axvline(
        x=np.mean(df["utilization_rate"]),
        color="#FF2800",
        linestyle="dashdot",
        linewidth=2,
        label="Average Utilization Line"
    )
    plt.xlabel("Utilization Rate")
    plt.ylabel("Number of MPs")
    plt.title("Distribution of MPLADS Fund Utilization Rate")
    plt.legend()
    plt.show()


def plot_state_boxplot(df: pd.DataFrame) -> None:
    sns.boxplot(data=df, x="state", y="utilization_rate", color="royalblue")
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("State", color="#FF2800")
    plt.ylabel("Utilization Rate", color="#FF2800")
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    corr_matrix = df[["allocated_amount", "total_sanction_amount",
                       "total_disbursed_amount", "utilization_rate"]].corr()
    sns.heatmap(corr_matrix, annot=True)
    plt.title("Correlation Heatmap: MPLADS Fund Metrics")
    plt.show()
    # Low correlation between allocated_amount and total_disbursed_amount
    # → more funds allocated does NOT mean more funds get utilized/completed.
    # Confirms zero-activity MP finding: money alone isn't the bottleneck.


def plot_top_states_bar(df: pd.DataFrame) -> None:
    state_total = (df.groupby("state")["allocated_amount"].sum()
                   .divide(1e7).sort_values(ascending=False).head(10).reset_index())
    ax = sns.barplot(
        data=state_total, x="allocated_amount", y="state",
        hue="allocated_amount", palette="ch:start=.2,rot=-.3", legend=False
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=5)
    plt.xlabel("Allocated Amount (in crores)")
    plt.ylabel("States")
    plt.title("States with the highest amount allocation")
    plt.tight_layout()
    plt.show()


def plot_zero_activity_scatter(df: pd.DataFrame) -> None:
    sns.scatterplot(
        data=df, x="allocated_amount", y="total_disbursed_amount",
        hue="has_no_activity", style="has_no_activity"
    )
    plt.xlabel("Allocated Amount")
    plt.ylabel("Total Disbursed Amount")
    plt.title("Amount Allocated vs Total Amount Disbursed")
    plt.legend(title="Has no activity")
    plt.show()


def plot_dashboard(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    plt.style.use("bmh")

    axes[0, 0].hist(df["utilization_rate"], bins=15, color="royalblue", edgecolor="black")
    axes[0, 0].axvline(
        x=np.mean(df["utilization_rate"]),
        color="#FF2800", linestyle="dashdot", linewidth=2,
        label="Average Utilization Line"
    )
    axes[0, 0].set_xlabel("Utilization Rate")
    axes[0, 0].set_ylabel("Number of MPs")
    axes[0, 0].set_title("Distribution of MPLADS Fund Utilization Rate")
    axes[0, 0].legend()

    sns.boxplot(data=df, x="state", y="utilization_rate", color="royalblue", ax=axes[0, 1])
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45, ha='right')
    axes[0, 1].set_xlabel("State", color="#FF2800")
    axes[0, 1].set_ylabel("Utilization Rate", color="#FF2800")

    corr_matrix = df[["allocated_amount", "total_sanction_amount",
                       "total_disbursed_amount", "utilization_rate"]].corr()
    sns.heatmap(corr_matrix, annot=True, ax=axes[0, 2])
    axes[0, 2].set_title("Correlation Heatmap: MPLADS Fund Metrics")

    state_total = (df.groupby("state")["allocated_amount"].sum()
                   .divide(1e7).sort_values(ascending=False).head(10).reset_index())
    sns.barplot(
        data=state_total, x="allocated_amount", y="state",
        hue="allocated_amount", palette="ch:start=.2,rot=-.3", legend=False,
        ax=axes[1, 0]
    )
    for container in axes[1, 0].containers:
        axes[1, 0].bar_label(container, fmt="%.2f", padding=5)
    axes[1, 0].set_xlabel("Allocated Amount (in crores)")
    axes[1, 0].set_ylabel("States")
    axes[1, 0].set_title("States with the highest amount allocation")

    sns.scatterplot(
        data=df, x="allocated_amount", y="total_disbursed_amount",
        hue="has_no_activity", style="has_no_activity", ax=axes[1, 1]
    )
    axes[1, 1].set_xlabel("Allocated Amount")
    axes[1, 1].set_ylabel("Total Disbursed Amount")
    axes[1, 1].set_title("Amount Allocated vs Total Amount Disbursed")
    axes[1, 1].legend(title="Has no activity")

    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = build_final_dataframe()

    plot_utlization_rate(df)
    plot_state_boxplot(df)
    plot_correlation_heatmap(df)
    plot_top_states_bar(df)
    plot_zero_activity_scatter(df)
    plot_dashboard(df)