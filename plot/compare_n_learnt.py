from typing import Iterable

import matplotlib.axis
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def rename_teachers(data: pd.DataFrame):
    dic = {
        "leitner": "Leitner",
        "forward": "Conservative\nsampling",
        "threshold": "Myopic",
    }

    for k, v in dic.items():
        data["teacher"] = data["teacher"].replace([k], v)
    return data


def boxplot_n_learnt(data: pd.DataFrame,
                     ax: matplotlib.axes._axes.Axes = None,
                     ylim: Iterable = None,
                     x_label: str = "Teacher",
                     y_label: str = "Learned",
                     dot_size: int = 3,
                     dot_alpha: float = 0.7):

    if ax is None:
        fig, ax = plt.subplots()

    data = rename_teachers(data)
    data = data.rename(columns={
        "n_learnt": y_label,
        "teacher": x_label
    })

    order = ["Leitner", "Myopic", "Conservative\nsampling"]
    colors = ["C0", "C1", "C2"]

    sns.boxplot(x=x_label, y=y_label, data=data, ax=ax,
                palette=colors, order=order,
                showfliers=False)
    sns.stripplot(x=x_label, y=y_label, data=data, s=dot_size,
                  color="0.25", alpha=dot_alpha, ax=ax, order=order)

    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=13)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xlabel("")
