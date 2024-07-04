import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import streamlit as st
from tabulate import tabulate
import io
import base64

def analyze_data(data, groups):
    df = pd.DataFrame(data.T, columns=groups * 3)

    normalized_values = []
    for i in range(0, len(groups) * 3, 3):
        avg_first_row = df.iloc[:, i].mean()
        for j in range(3):
            normalized_values.append(df.iloc[:, i + j] / avg_first_row)

    all_normalized_values = []
    group_labels = []
    for i in range(len(groups)):
        for j in range(i, len(normalized_values), 3):
            valid_values = normalized_values[j].dropna()
            all_normalized_values.extend(valid_values)
            group_labels.extend([groups[i]] * len(valid_values))

    anova_df = pd.DataFrame({'value': all_normalized_values, 'group': group_labels})

    model = ols('value ~ C(group)', data=anova_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    tukey = pairwise_tukeyhsd(endog=anova_df['value'], groups=anova_df['group'], alpha=0.05)
    significant_pairs = tukey.reject

    means = []
    std_devs = []

    for group in groups:
        group_values = anova_df[anova_df['group'] == group]['value']
        means.append(np.mean(group_values))
        std_devs.append(np.std(group_values))

    return anova_df, anova_table, tukey, significant_pairs, means, std_devs

def plot_results(groups, anova_df, tukey, significant_pairs, means, std_devs):
    def add_significance(ax, x1, x2, y, h, text):
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, color='black')
        ax.text((x1 + x2) * .5, y + h, text, ha='center', va='bottom', color='black', fontsize=12)

    # Bar plot
    fig, ax = plt.subplots(figsize=(8, 8))
    bars = ax.bar(groups, means, yerr=std_devs, capsize=10, color='#88c7dc')

    ax.set_title('Comparison of Group Means', fontsize=15)
    ax.set_ylabel('Mean Values', fontsize=12)

    if np.any(significant_pairs):
        max_val = max(means) + max(std_devs)
        h = max_val * 0.05
        gap = max_val * 0.02
        whisker_gap = max_val * 0.02

        comparisons = np.array(tukey.summary().data[1:])
        significant_comparisons = comparisons[significant_pairs]

        for comp in significant_comparisons:
            if 'siRNA_ctrl' in comp[:2]:
                group1 = groups.index(comp[0])
                group2 = groups.index(comp[1])
                add_significance(ax, group1, group2, max_val + whisker_gap, h, '*')
                whisker_gap += h + gap

    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.grid(False)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode()

    plt.close(fig)

    return plot_url

def display_table(anova_table, tukey):
    anova_table_html = anova_table.to_html(classes='table table-striped')
    tukey_summary_html = tukey.summary().as_html()
    return anova_table_html, tukey_summary_html

st.title('ANOVA Analysis')

data_input = st.text_area('Data (numpy array format):', '[[998656.0, 1084988.0, 916308.0, 722976.0, 962084.0], [604008.0, 777348.0, 791404.0, 552608.0, 685984.0], [736832.0, 818552.0, 1055040.0, 1287956.0, 1172576.0], [1662848.0, 1834048.0, 1833852.0, 1693920.0, 1436480.0], [1251460.0, 1413120.0, 1318736.0, 1323892.0, 1276216.0], [1443384.0, 1572480.0, 1397224.0, 1482584.0, 1345784.0], [1274324.0, 1269692.0, 1312584.0, 1287352.0, 1366100.0], [1089768.0, 996400.0, 1111588.0, 1031416.0, 1071800.0], [1207848.0, 1123748.0, 1296440.0, 1266744.0, 1212152.0]]')
groups_input = st.text_area('Groups (list format):', "['siRNA_ctrl', 'siRNA1_VTN', 'siRNA2_VTN']")

if st.button('Run Analysis and Plot'):
    data = np.array(eval(data_input))
    groups = eval(groups_input)

    anova_df, anova_table, tukey, significant_pairs, means, std_devs = analyze_data(data, groups)
    anova_table_html, tukey_summary_html = display_table(anova_table, tukey)
    plot_url = plot_results(groups, anova_df, tukey, significant_pairs, means, std_devs)

    st.markdown(anova_table_html, unsafe_allow_html=True)
    st.markdown(tukey_summary_html, unsafe_allow_html=True)
    st.image(f"data:image/png;base64,{plot_url}")
