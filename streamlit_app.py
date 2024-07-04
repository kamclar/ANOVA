import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols, wls
import streamlit as st
from tabulate import tabulate
import io
import base64
from io import StringIO
import scikit_posthocs as sp

def analyze_weighted_anova(data, groups):
    df = pd.DataFrame(data.T, columns=groups * 3)

    normalized_values = []
    for i in range(0, len(groups) * 3, 3):
        avg_first_row = df.iloc[:, i].mean()
        for j in range(3):
            normalized_values.append(df.iloc[:, i + j] / avg_first_row)

    all_normalized_values = []
    group_labels = []
    weights = []
    for i in range(len(groups)):
        for j in range(i, len(normalized_values), 3):
            valid_values = normalized_values[j].dropna()
            row_length = len(valid_values)
            weight = 1 / row_length
            all_normalized_values.extend(valid_values)
            group_labels.extend([groups[i]] * row_length)
            weights.extend([weight] * row_length)

    total_observations = len(all_normalized_values)
    weights = [w * total_observations / sum(weights) for w in weights]

    anova_df = pd.DataFrame({'value': all_normalized_values, 'group': group_labels, 'weights': weights})

    model = wls('value ~ C(group)', data=anova_df, weights=anova_df['weights']).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    means = []
    std_devs = []

    for group in groups:
        group_values = anova_df[anova_df['group'] == group]['value']
        means.append(np.mean(group_values))
        std_devs.append(np.std(group_values))

    return anova_df, anova_table, means, std_devs, "Weighted ANOVA"

def analyze_standard_anova(data, groups):
    df = pd.DataFrame(data.T, columns=groups * 3)

    melted_df = pd.melt(df.reset_index(), id_vars=['index'], value_vars=groups * 3)
    melted_df.columns = ['index', 'group', 'value']

    model = ols('value ~ C(group)', data=melted_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    means = []
    std_devs = []

    for group in groups:
        group_values = melted_df[melted_df['group'] == group]['value']
        means.append(np.mean(group_values))
        std_devs.append(np.std(group_values))

    return melted_df, anova_table, means, std_devs, "Standard ANOVA"

def dunnett_test(anova_df, control_group):
    comp = sp.posthoc_dunn(anova_df, val_col='value', group_col='group', p_adjust='bonferroni')
    control_comp = comp.loc[control_group]
    return control_comp

def plot_results(groups, anova_df, dunnett_results, means, std_devs, analysis_type):
    fig, ax = plt.subplots()

    x = np.arange(len(groups))
    ax.bar(x, means, yerr=std_devs, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Values')
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_title(f'{analysis_type} Results')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    plot_url = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return plot_url

def display_table(anova_table, dunnett_results):
    anova_table_html = anova_table.to_html(classes='table table-striped')
    dunnett_html = dunnett_results.to_frame().to_html(classes='table table-striped')
    return anova_table_html, dunnett_html

def parse_pasted_data(pasted_data, delimiter):
    # Split the data into lines
    lines = pasted_data.strip().split("\n")
    # Split each line into columns
    data = [line.split(delimiter) for line in lines]
    # Find the maximum number of columns
    max_cols = max(len(row) for row in data)
    # Pad the rows to have the same number of columns
    padded_data = [row + [''] * (max_cols - len(row)) for row in data]
    # Convert to DataFrame
    df = pd.DataFrame(padded_data).replace('', np.nan)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

st.title('ANOVA Analysis')

delimiter = st.selectbox('Select delimiter', ('\t', ';', ','))

input_method = st.radio("Select input method", ('Copy-Paste', 'File Upload', ))

if input_method == 'File Upload':
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
else:
    pasted_data = st.text_area("Paste your data here (use selected delimiter)")

if (input_method == 'File Upload' and uploaded_file is not None) or (input_method == 'Copy-Paste' and pasted_data):
    try:
        if input_method == 'File Upload':
            data = pd.read_csv(uploaded_file, delimiter=delimiter, header=None)
        else:
            data = parse_pasted_data(pasted_data, delimiter)

        st.write("Data Preview:", data.head())

        data_values = data.values
        st.text_area('Data (numpy array format):', str(data_values))

        groups_input = st.text_area('Groups (list format):', "['siRNA_ctrl', 'siRNA1_VTN', 'siRNA2_VTN']")

        if st.button('Run Analysis and Plot'):
            groups = eval(groups_input)

            # Determine which ANOVA to use
            if data.isnull().values.any():
                anova_df, anova_table, means, std_devs, analysis_type = analyze_weighted_anova(data_values, groups)
            else:
                anova_df, anova_table, means, std_devs, analysis_type = analyze_standard_anova(data_values, groups)

            dunnett_results = dunnett_test(anova_df, groups[0])

            st.write(f"Analysis Type: {analysis_type}")

            anova_table_html, dunnett_html = display_table(anova_table, dunnett_results)
            plot_url = plot_results(groups, anova_df, dunnett_results, means, std_devs, analysis_type)

            st.markdown(anova_table_html, unsafe_allow_html=True)
            st.markdown(dunnett_html, unsafe_allow_html=True)
            st.image(f"data:image/png;base64,{plot_url}")
    except Exception as e:
        st.error(f"Error processing the file: {e}")
