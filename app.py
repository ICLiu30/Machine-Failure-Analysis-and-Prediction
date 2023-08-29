import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

st.set_option('deprecation.showPyplotGlobalUse', False)
dir = Path(__file__).resolve().parent  
###
target = 'Machine failure'
failure_category = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
numeric = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
           'Torque [Nm]', 'Tool wear [min]']
###

###
def plot_kde_distribution(df, features, target, num_cols):
    num_rows = (len(features) + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    flattened_axes = axes.flatten()

    for i, ax in enumerate(flattened_axes):
        if i < len(features):
            sns.kdeplot(data=df[df[target] == 0], x=features[i], fill=True, linewidth=2, ax=ax, label='Without Failure', color='#F3DBD3')
            sns.kdeplot(data=df[df[target] == 1], x=features[i], fill=True, linewidth=2, ax=ax, label='With Failure', color='#F5875F')

            ax.set_yticks([])
            ax.set_xlabel(f'{features[i]}', fontsize=14)
            ax.set_ylabel('Density', fontsize=14)

            ax.legend(title=target, loc='upper right')
        else:
            ax.axis('off')

    plt.suptitle('Numeric Feature Distributions vs ' + target, fontsize=22, y=1.02)
    plt.tight_layout()
    return 

def plot_single_kde(df, feature, target):
    plt.figure(figsize=(15,8))
    
    sns.kdeplot(data=df[df[target] == 0], x=feature, fill=True, linewidth=2, label='Without Failure', color='#F3DBD3')
    sns.kdeplot(data=df[df[target] == 1], x=feature, fill=True, linewidth=2, label='With Failure', color='#F5875F')
    plt.yticks([])
    plt.xlabel(feature, fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.legend(title=target, loc='upper right')
    plt.title(f'KDE Distribution of {feature} by {target}', fontsize=18)
    return plt.gcf() 

###

###
path_to_SHAP_x = dir / 'data' / 'SHAP_x.csv'
path_to_SHAP_y = dir / 'data' / 'SHAP_y.csv'
path_to_importance = dir / 'data' / 'importance_df.csv'
path_to_data = dir / 'data' / 'Plot.csv'
path_to_SHAP_png = dir / 'data' / 'SHAP.png'
path_to_Matrix_png = dir / 'data' / 'ConfusionMatrix.png'
###

###
df = pd.read_csv(path_to_data)
df_importance = pd.read_csv(path_to_importance)
###
color = ['#F3DBD3', '#F5875F']
###
st.title('Machine Failures Analysis')
st.write("""
This project is inspired by a Kaggle competition, which can be explored [here](https://www.kaggle.com/competitions/playground-series-s3e17). The primary objective is to understand potential factors contributing to machine failures and to predict such failures using the available features:

- **Product ID**: A categorical attribute distinguishing product types.
- **Air temperature (K)**: The environmental temperature.
- **Process temperature (K)**: Temperature associated with the production process.
- **Rotational speed (rpm)**: Indicates the speed of the main shaft's rotation.
- **Torque (Nm)**: Typically around 40 Nm with a variability (ε) of 10 and devoid of negative values.
- **Tool wear (min)**: Represents the operation duration of the tool.

The dataset also presents six specific equipment failure types:
- **Tool wear failure (TWF)**: A failure caused due to tool wear.
- **Heat dissipation failure (HDF)**: Failure due to inadequate heat dissipation.
- **Power failure (PWF)**: Indicates a process breakdown caused by power failure.
- **Overstrain failure (OSF)**: Pertains to failure because of excessive strain during production.
- **Random failures (RNF)**: Failures with indeterminate causes.
- **Machine failure**: A binary label where 0 denotes normal operation and 1 signifies a failure.

By leveraging this data, the goal is to provide actionable insights for preventive machine maintenance and operation optimization.
""")



st.write("## Exploratory Data Analysis")


st.write("### Proportions of Machine Failures")
st.write('I began by examining the proportion of failed machines relative to the entire set of machines. Evidently, only a small fraction (1.6%) represented machine failures.')
target_proportions = df[target].value_counts() / len(df)

label_map = {0: 'Operational', 1: 'Failed'}
labels_failure = [label_map[idx] for idx in target_proportions.index.tolist()]

sizes_failure = target_proportions.values.tolist()
explode_failure = [0, 0.05]

fig_failure, ax1 = plt.subplots(figsize=(15,8))
ax1.pie(sizes_failure, labels = None, autopct='%1.1f%%', startangle=70,
        shadow=False, pctdistance=1.13, explode=explode_failure, colors=color, textprops={'fontsize': 12})

ax1.set_title('Proportions of Machine Failure', y=1.05, fontsize=20)
ax1.legend(labels = labels_failure, loc='upper left', fontsize=14)
ax1.axis('equal')

st.pyplot(fig_failure)

st.write('### Analysis of Categorical Features')
st.write('#### Machine Type Analysis and Its Impact on Failure Rates')
st.write("I identified three machine types: L, M, and H. Most machines are of type L. Upon analyzing the failure rate for each type, it's clear that type L machines have the highest failure ratio. However, with failure rate differences being under 10% across types, machine type doesn't appear to be a major factor in failures.")

type_proportions = df['Type'].value_counts() / len(df)

labels_type = type_proportions.index.tolist()
sizes_type = type_proportions.values.tolist()

color_palette = {'L': '#CCE5FF', 'M': '#66B2FF', 'H': '#0080FF'}
colors_type = [color_palette[label] for label in labels_type]
explode_type = [0, 0, 0.1]

fig_type, ax2 = plt.subplots(figsize=(15,8))
ax2.pie(sizes_type, labels = None, autopct='%1.1f%%', startangle=70,
        shadow=False, pctdistance=1.13, explode=explode_type, textprops={'fontsize': 12}, colors=colors_type)

ax2.set_title('Proportions of Type of Machine', y=1.05, fontsize=20)
ax2.legend(labels = labels_type, loc='upper left', fontsize=14)
ax2.axis('equal')

st.pyplot(fig_type)

fig_type_failure, axes = plt.subplots(1, 3, figsize=(15,10))
flattened_axes = axes.flatten()

Type = ['L', 'M', 'H']
explode = [0, 0.05]

for i, ax in enumerate(flattened_axes):

    mask = (df['Type'] == Type[i])
    data = df[mask]
    target_proportions = data['Machine failure'].value_counts() / len(data)
    
    label_map = {0: 'Without Failure', 1: 'With Failure'}
    labels = [label_map[idx] for idx in target_proportions.index.tolist()]
    sizes = target_proportions.values.tolist()

    ax.pie(sizes, labels = None, autopct='%1.1f%%', startangle=70,
           pctdistance=1.13, explode=explode, colors=color, textprops={'fontsize': 14})
    
    ax.set_title(f'Proportions of Machine Failure of {Type[i]}', y=1.05, fontsize=16)
    ax.legend(labels = labels, loc='upper left', fontsize=12)

plt.tight_layout()

st.pyplot(fig_type_failure)

st.write("#### Failure Mode Analysis")
st.write("Exploring the proportions of failed machines across different failure modes: TWF, HDF, PWF, OSF, and RNF reveals that machines with HDF, OSF, or PWF modes are more prone to failures.")

fig_failuremode, axes = plt.subplots(3, 2, figsize=(15,15))

flattened_axes = axes.flatten()

data = df[df['Machine failure'] == 1]

explode = [0, 0.05]

for i, ax in enumerate(flattened_axes):
    if i < 5:

        target_proportions = data[failure_category[i]].value_counts() / len(data)
        label_map = {0: f'Without {failure_category[i]}', 1: f'With {failure_category[i]}'}
        labels = [label_map[idx] for idx in target_proportions.index.tolist()]

        sizes = target_proportions.values.tolist()

        ax.pie(sizes, labels = None, autopct='%1.1f%%', startangle=70,
               shadow=False, pctdistance=1.2, explode=explode, colors=color, textprops={'fontsize': 12})

        ax.set_title(f'Proportions of Machines with {failure_category[i]}', y=1.05, fontsize=14)
        ax.legend(labels = labels, loc='upper left', fontsize=10)
    else:
        ax.axis('off')

plt.tight_layout()

st.pyplot(fig_failuremode)

st.write('### Analysis of Numeric Features')

st.write("#### Analysis of Numeric Feature Distributions")
st.write("Upon examining the numeric features' histogram, categorized by machine type, it's evident that distributions across machine types are quite similar. This confirms that machine type doesn't have a strong influence on the likelihood of failure. Additionally, most numeric features exhibit a near-normal distribution, showing no significant skewness.")


fig_numeric, axes = plt.subplots(3, 2, figsize=(15, 15))
flattened_axes = axes.flatten()
desired_order = ['L', 'M', 'H']

for i, ax in enumerate(flattened_axes):
    
    if i < len(numeric):  
        feature = numeric[i]
        sns.histplot(data=df, x=feature, kde=False, hue='Type', hue_order=desired_order, ax=ax, palette=color_palette)
        ax.set_xlabel(f'{feature}', fontsize=14)
        ax.set_ylabel('Counts', fontsize=14)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

    else:
        ax.axis('off')
    
plt.tight_layout()

st.pyplot(fig_numeric)

st.write("#### Violin Plots: Numeric Feature Insights")

st.write("""
The violin plots illustrate the density of our machine failure targets for each numeric feature, segregated by machine type. The wider the plot's area, the higher the density of the target in relation to a particular feature value.

Key observations from the plots:
- For "Rotational speed", machines operating below approximately 1500 rpm are more prone to failures.
- For "Torque", values higher than 45 Nm indicate an increased likelihood of machine failure.
- For "Air temperature", readings above 302 K are associated with a greater risk of machine malfunction.

Though the distributions across different machine types are largely consistent, the divergence in target distributions within features like "Rotational speed", "Air temperature", and "Torque" implies their potential significance in influencing machine failure. Nonetheless, it's essential to consider potential interactions among features, as they may also be influential in determining machine reliability. We will delve into these interactions in subsequent sections.
""")


fig_violin, axes = plt.subplots(3, 2, figsize=(15, 15))
    
flattened_axes = axes.flatten()
    
for i, ax in enumerate(flattened_axes):
    if i < len(numeric):
        sns.violinplot(x='Type', y=numeric[i], hue='Machine failure', split=True, data=df, ax=ax, palette=color, order=["L", "M", "H"], hue_order=[0,1])
        
        ax.set_xlabel('Type', fontsize=14)
        ax.set_ylabel(f'{numeric[i]}', fontsize=14)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
            
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, ["Operational", "Failed"], loc='upper left', fontsize=10)
            
    else:
        ax.axis('off')
    
plt.tight_layout()

st.pyplot(fig_violin)

st.write('### Advamced Analysis of Failure Mode')
st.write("""
Let's delve deeper into the analysis of each failure mode to determine if specific features contribute to certain failures. 
From our pie charts, we observe that machines with HDF, OSF, or PWF have the most pronounced failure rates. Thus, 
our primary attention will be on these three failure modes. 

For this section, we employ KDE (Kernel Density Estimation) plots for our analysis. A KDE plot, much like the violin plot, 
visualizes the distribution of data over a continuous interval. In these plots, larger areas signify a higher density of data points.

To tailor the analysis to your interests, use the select box to choose the specific failure mode you wish to examine.
""")


# Failure analysis
mode = st.selectbox('Mode', failure_category)

if mode == 'TWF':
    st.write('#### Tool Wear Failure Analysis')
    st.write("""
            From the KDE plot, 'Tool wear [min]' emerges as the predominant feature influencing TWF. This insight aligns well with intuitive expectations.
            """)
    fig_TWF = plot_kde_distribution(df, numeric, 'TWF', 2)
    st.pyplot(fig_TWF)

elif mode == 'HDF':
    st.write('#### Heat Dissipation Failure Analysis')
    st.write("""
            The KDE plot highlights 'Air temperature [K]' and 'Process temperature [K]' as key influencers of HDF. Given thermodynamics principles, heat dissipation largely depends on the temperature difference between the system and the environment. Thus, I've added a 'Temperature difference [K]' feature to further explore its impact on HDF.
            """)

    fig_HDF = plot_kde_distribution(df, numeric, 'HDF', 2)
    st.pyplot(fig_HDF)

    st.write("""
            The KDE distribution reveals a stark differentiation between machines with and without HDF based on the new feature. This confirms our initial hypothesis.
            """)
    fig_HDF_new = plot_single_kde(df,'Temperature difference [K]', 'HDF')

    st.pyplot(fig_HDF_new)

elif mode == 'PWF':
    st.write('#### Power Failure Analysis')
    st.write("""
            Based on the KDE plot, 'Torque [Nm]' and 'Rotational speed [rpm]' prominently influence PWF. Given that power in mechanical systems is the product of torque and speed, I introduced a new feature: 'Power' = Torque [Nm] × Rotational speed [rpm].
            """)

    fig_PWF = plot_kde_distribution(df, numeric, 'PWF', 2)
    st.pyplot(fig_PWF)
    st.write("""
            The KDE distribution showcases the new 'Power' feature's ability to distinctly differentiate between machines with and without PWF, validating our hypothesis.
            """)

    fig_PWF_new = plot_single_kde(df,'Power [Nm*rpm]', 'PWF')
    st.pyplot(fig_PWF_new)

elif mode == 'OSF':
    st.write('#### Overstrain Failure Analysis')
    st.write("""
            The KDE plot highlights 'Torque [Nm]' and 'Tool wear [min]' as key determinants for OSF. A potential interplay exists: higher torque might hasten tool wear, while worn-out tools could demand more torque. Hence, I introduced an interaction feature, 'Torque Tool Wear' = Torque [Nm] × Tool wear [min].
            """)

    fig_OSF = plot_kde_distribution(df, numeric, 'OSF', 2)
    st.pyplot(fig_OSF)

    st.write("""
            The KDE distribution underscores the distinctiveness of the 'Torque Tool Wear' feature in differentiating machines with and without OSF, validating our hypothesis.
            """)

    fig_OSF_new = plot_single_kde(df,'Torque Toll Wear [Nm*min]', 'OSF')
    st.pyplot(fig_OSF_new)

elif mode == 'RNF':
    st.write('#### Random Failure Analysis')
    st.write("The KDE distribution reveals no distinct feature differences for machines with a random failure mode.")

    fig_RNF = plot_kde_distribution(df, numeric, 'RNF', 2)
    st.pyplot(fig_RNF)

## Feature importance
st.write('#### Feature Importance and SHAP Value Analysis')

st.write("""
         ### Feature Importance

        Feature importance gives a measure of how influential each feature is in the model's prediction. In XGBoost, it's determined by the number of times a feature appears in a tree across multiple trees. By understanding which features play a critical role and which don't, we can streamline our model and data.
        """)

df_importance_sorted = df_importance.sort_values(by='importance', ascending=True)

# Create the horizontal bar chart using Plotly
fig_importance = px.bar(df_importance_sorted,
                        x='importance',
                        y='features',
                        orientation='h',  # This makes the bar chart horizontal
                        title='Feature Importance',
                        labels={'importance': 'Importance', 'features': 'Features'}
                        )

st.plotly_chart(fig_importance)

# SHAP
st.write("""
        #### Understanding Feature Importance with SHAP

        Originating from game theory, SHAP (SHapley Additive exPlanations) values offer a unified measure of feature importance, elucidating how each feature influences a particular prediction. Unlike traditional metrics that merely indicate the importance of a feature, SHAP values specify the direction of that feature's impact — positive SHAP values push the model's output up, while negative ones drag it down. This depth of interpretation grants insights into feature interactions and their combined effects.

        In the context of our machine failure prediction model, a higher SHAP value for a feature suggests that the feature's presence or value makes the machine more susceptible to failure. As the SHAP value for a specific feature rises, the machine's risk of being identified as "prone to failure" by the model also increases. This dual insight is invaluable, both for gauging individual feature importance and understanding potential risk factors associated with machine failures.
        """)

st.image(str(path_to_SHAP_png), use_column_width=True)

st.write("""
        #### Feature Selection based on Feature Importance and SHAP Values

        From the XGBoost feature importance and SHAP value plots shown above, we gain insight into how features influence predictions. Positive SHAP values indicate a feature pushes the model's prediction towards being more prone to failure.

        Using this understanding, I've opted to remove the features Type_H, Type_L, Type_M, RNF, Process temperature K. By doing so, the model becomes simpler, offering clearer interpretations while retaining adaptability to varied situations.
        """)


# SHAP Analysis
df_SHAP_x = pd.read_csv(path_to_SHAP_x, index_col=0)
df_SHAP_y = pd.read_csv(path_to_SHAP_y, index_col=0)


st.write("""
### SHAP Scatter Plots: Understanding Feature Influence

Using the select box below, you can choose and delve into the feature you're interested in. 

For instance, the plot for 'Air temperature [K]' illustrates that values higher than 302 K typically have a SHAP value exceeding 0, indicating a heightened likelihood of machine failure. This pattern holds true for other features as well. With these plots, you can assess if your machine operates in a 'dangerous zone' (where SHAP value > 1) and take preventive measures accordingly.
""")

feature = st.selectbox('Feature', df_SHAP_x.columns.to_list())

fig_scatter = px.scatter(x=df_SHAP_x[feature], 
                         y=df_SHAP_y[feature], 
                         title=f'SHAP value of {feature}',
                         labels={'x': feature, 'y': 'SHAP Value'},
                         color_discrete_sequence=["blue"])

fig_scatter.add_shape(type="line",
                      x0=min(df_SHAP_x[feature]), x1=max(df_SHAP_x[feature]),
                      y0=0, y1=0,
                      line=dict(color="Red", width=2, dash="dot")
                     )

fig_scatter.update_layout(font=dict(size=16, color="DarkBlue"))

st.plotly_chart(fig_scatter)

# Confusion matrix
st.write("""
### Conclusion

Through rigorous analysis, I uncovered crucial insights into the various factors influencing machine failures. By leveraging the XGBoost model, I achieved an impressive ROC_AUC score of 0.94, emphasizing the model's ability to effectively distinguish between operational and failed machines.

A closer look at the confusion matrix reveals:
- **True Positives:** 180 machines correctly identified as 'Failed'.
- **True Negatives:** 13,007 machines accurately labeled as 'Operational'.
- **False Positives:** 421 operational machines mistakenly predicted as 'Failed'.
- **False Negatives:** 35 failed machines erroneously classified as 'Operational'.

From this matrix, the following metrics emerged:
- **Recall (Sensitivity):** Approximately 0.837, suggesting that the model identified 83.7% of all actual machine failures.
- **Precision:** Approximately 0.299, indicating that of all the machines the model labeled as 'Failed', 29.9% truly were.

The high recall is particularly significant, reflecting the model's capacity to detect most of the machine failures, which is vital for timely interventions and maintenance.

In conclusion, these findings are instrumental in promoting the adoption of predictive maintenance systems, ensuring increased safety and operational efficiency. The actionable insights from the SHAP values can guide industries to act proactively, especially when machines venture into the 'dangerous zones' of operation.
""")


st.image(str(path_to_Matrix_png), use_column_width=True)