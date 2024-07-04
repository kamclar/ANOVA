import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols, mixedlm
import streamlit as st
from tabulate import tabulate
import io
import base64
from io import StringIO

def analyze_mixed_effects(data, groups):
    biological_replicates = len(groups)
    technical_replicates = data.shape[1] // biological_replicates

    df = pd.DataFrame(data.T, columns=groups * biological_replicates)

    normalized_values = []
    for i in range(0, len(groups) * biological_replicates, biological_replicates):
        avg_first_row = df.iloc[:, i].mean()
        for j in range(biological_replicates):
            normalized_values.append(df.iloc[:, i + j] / avg_first_row)

    all_normalized_values = []
    group_labels = []
    subject_labels = []
    for i in range(len(groups)):
        for j in range(i, len(normalized_values), biological_replicates):
            valid_values = normalized_values[j].dropna()
            all_normalized_values.extend(valid_values)
            group_labels.extend([groups[i]] * len(valid_values))
            subject_labels.extend([f'subject_{j // biological_replicates}'] * len(valid_values))

    anova_df = pd.DataFrame({'value': all_normalized_values, 'group': group_labels, 'subject': subject_labels})

    model = mixedlm('value ~ C(group)', anova_df, groups=anova_df['subject'])
    result = model.fit()

    means = []
    std_devs = []

    for group in groups:
        group_values = anova_df[anova_df['group'] == group]['value']
        means.append(np.mean(group_values))
        std_devs.append(np.std(group_values))

    return anova_df, result.summary(), means, std_devs, "Mixed-Effects Model"

def plot_results(groups, anova_df, mixed_model_results, means, std_devs, analysis_type):
    def add_significance(ax, x1, x2, y, h, text):
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, color='black')
        ax.text((x1 + x2) * .5, y + h, text, ha='center', va='bottom', color='black', fontsize=12)

    # Bar plot
    fig, ax = plt.subplots(figsize=(8, 8))
    bars = ax.bar(groups, means, yerr=std_devs, capsize=10, color='#88c7dc')

    ax.set_title(f'Comparison of Group Means ({analysis_type})', fontsize=15)
    ax.set_ylabel('Mean Values', fontsize=12)

    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.grid(False)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode()

    plt.close(fig)

    return plot_url

def display_table(mixed_model_summary):
    mixed_model_html = mixed_model_summary.tables[1].as_html()
    return mixed_model_html

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

st.title('ANOVA Analysis with Mixed-Effects Model')

delimiter = st.selectbox('Select delimiter', (';', '\t', ','))

input_method = st.radio("Select input method", ('File Upload', 'Copy-Paste'))

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

            anova_df, mixed_model_summary, means, std_devs, analysis_type = analyze_mixed_effects(data_values, groups)

            st.write(f"Analysis Type: {analysis_type}")

            mixed_model_html = display_table(mixed_model_summary)
            plot_url = plot_results(groups, anova_df, mixed_model_summary, means, std_devs, analysis_type)

            st.markdown(mixed_model_html, unsafe_allow_html=True)
            st.image(f"data:image/png;base64,{plot_url}")
    except Exception as e:
        st.error(f"Error processing the file: {e}")
