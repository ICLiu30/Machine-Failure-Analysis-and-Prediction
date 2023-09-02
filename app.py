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


###
path_to_SHAP_x = dir / 'data' / 'SHAP_x.csv'
path_to_SHAP_y = dir / 'data' / 'SHAP_y.csv'
path_to_importance = dir / 'data' / 'importance_df.csv'

path_to_SHAP_png = dir / 'png' / 'SHAP.png'
path_to_Matrix_png = dir / 'png' / 'ConfusionMatrix.png'
path_to_FailureModeAnalysis_png = dir / 'png' / 'FailureModeAnalysis.png'
path_to_HDF_png = dir / 'png' / 'HDF.png'
path_to_MachineType1_png = dir / 'png' / 'MachineType1.png'
path_to_MachineType2_png = dir / 'png' / 'MachineType2.png'
path_to_NumericFeatureDistributions_png = dir / 'png' / 'NumericFeatureDistributions.png'
path_to_NumericViolin_png = dir / 'png' / 'NumericViolin.png'
path_to_OSF_png = dir / 'png' / 'OSF.png'
path_to_Power_png = dir / 'png' / 'Power.png'
path_to_ProportionsofMachineFailures_png = dir / 'png' / 'ProportionsofMachineFailures.png'
path_to_RNF_png = dir / 'png' / 'RNF.png'
path_to_TemperatureDifference_png = dir / 'png' / 'TemperatureDifference.png'
path_to_TorqueTollWear_png = dir / 'png' / 'TorqueTollWear.png'
path_to_TWF_png = dir / 'png' / 'TWF.png'
path_to_PWF_png = dir / 'png' / 'PWF.png'

###

###
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

st.write("""
---

### Note to Readers:

Due to the large dataset, rendering interactive plots for every visualization would significantly slow down the deployment process online. To ensure a smoother experience, some plots are presented as PNG figures. Apologies for any inconvenience, and thank you for your understanding!

---
""")


st.write("## Exploratory Data Analysis")


st.write("### Proportions of Machine Failures")
st.write('I began by examining the proportion of failed machines relative to the entire set of machines. Evidently, only a small fraction (1.6%) represented machine failures.')

st.image(str(path_to_ProportionsofMachineFailures_png), use_column_width=True)

st.write('### Analysis of Categorical Features')
st.write('#### Machine Type Analysis and Its Impact on Failure Rates')
st.write("I identified three machine types: L, M, and H. Most machines are of type L. Upon analyzing the failure rate for each type, it's clear that type L machines have the highest failure ratio. However, with failure rate differences being under 10% across types, machine type doesn't appear to be a major factor in failures.")

st.image(str(path_to_MachineType1_png), use_column_width=True)
st.write('')
st.image(str(path_to_MachineType2_png), use_column_width=True)

st.write("#### Failure Mode Analysis")
st.write("Exploring the proportions of failed machines across different failure modes: TWF, HDF, PWF, OSF, and RNF reveals that machines with HDF, OSF, or PWF modes are more prone to failures.")

st.image(str(path_to_FailureModeAnalysis_png), use_column_width=True)

st.write('### Analysis of Numeric Features')

st.write("#### Analysis of Numeric Feature Distributions")
st.write("Upon examining the numeric features' histogram, categorized by machine type, it's evident that distributions across machine types are quite similar. This confirms that machine type doesn't have a strong influence on the likelihood of failure. Additionally, most numeric features exhibit a near-normal distribution, showing no significant skewness.")

st.image(str(path_to_NumericFeatureDistributions_png), use_column_width=True)

st.write("#### Violin Plots: Numeric Feature Insights")

st.write("""
The violin plots illustrate the density of our machine failure targets for each numeric feature, segregated by machine type. The wider the plot's area, the higher the density of the target in relation to a particular feature value.

Key observations from the plots:
- For "Rotational speed", machines operating below approximately 1500 rpm are more prone to failures.
- For "Torque", values higher than 45 Nm indicate an increased likelihood of machine failure.
- For "Air temperature", readings above 302 K are associated with a greater risk of machine malfunction.

Though the distributions across different machine types are largely consistent, the divergence in target distributions within features like "Rotational speed", "Air temperature", and "Torque" implies their potential significance in influencing machine failure. Nonetheless, it's essential to consider potential interactions among features, as they may also be influential in determining machine reliability. We will delve into these interactions in subsequent sections.
""")

st.image(str(path_to_NumericViolin_png), use_column_width=True)

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
    st.image(str(path_to_TWF_png), use_column_width=True)

elif mode == 'HDF':
    st.write('#### Heat Dissipation Failure Analysis')
    st.write("""
            The KDE plot highlights 'Air temperature [K]' and 'Process temperature [K]' as key influencers of HDF. Given thermodynamics principles, heat dissipation largely depends on the temperature difference between the system and the environment. Thus, I've added a 'Temperature difference [K]' feature to further explore its impact on HDF.
            """)
    
    st.image(str(path_to_HDF_png), use_column_width=True)

    st.write("""
            The KDE distribution reveals a stark differentiation between machines with and without HDF based on the new feature. This confirms our initial hypothesis.
            """)
    st.image(str(path_to_TemperatureDifference_png), use_column_width=True)

elif mode == 'PWF':
    st.write('#### Power Failure Analysis')
    st.write("""
            Based on the KDE plot, 'Torque [Nm]' and 'Rotational speed [rpm]' prominently influence PWF. Given that power in mechanical systems is the product of torque and speed, I introduced a new feature: 'Power' = Torque [Nm] × Rotational speed [rpm].
            """)

    st.image(str(path_to_PWF_png), use_column_width=True)

    st.write("""
            The KDE distribution showcases the new 'Power' feature's ability to distinctly differentiate between machines with and without PWF, validating our hypothesis.
            """)
    
    st.image(str(path_to_Power_png), use_column_width=True)

elif mode == 'OSF':
    st.write('#### Overstrain Failure Analysis')
    st.write("""
            The KDE plot highlights 'Torque [Nm]' and 'Tool wear [min]' as key determinants for OSF. A potential interplay exists: higher torque might hasten tool wear, while worn-out tools could demand more torque. Hence, I introduced an interaction feature, 'Torque Tool Wear' = Torque [Nm] × Tool wear [min].
            """)
    st.image(str(path_to_OSF_png), use_column_width=True)
    st.write("""
            The KDE distribution underscores the distinctiveness of the 'Torque Tool Wear' feature in differentiating machines with and without OSF, validating our hypothesis.
            """)
    
    st.image(str(path_to_TorqueTollWear_png), use_column_width=True)

elif mode == 'RNF':
    st.write('#### Random Failure Analysis')
    st.write("The KDE distribution reveals no distinct feature differences for machines with a random failure mode.")

    st.image(str(path_to_NumericViolin_png), use_column_width=True)

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
