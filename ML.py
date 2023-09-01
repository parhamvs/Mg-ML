import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sn
import shap

# Load data from an Excel file
df = pd.read_excel('Mg-GA-Added.xlsx')

# Define columns for different groups of elements
element_columns = [
    'Sr2Mg17(s)', 'Sr5Si3(s)', 'SrZn5(s)', 'Sr2Zn43Mg55(s)',
    'Mg12Zn13(s)', 'Ca2Mg55Zn43(s)', 'SrZn5(s)', 'Sr2Zn43Mg55(s)',
    'YZnMg12(s)', 'Zn2Zr(s)', 'ZnZr(s)', 'Nd3Zn22(s)', 'NdZn2Mg(s)',
    'Nd2Zn9Mg5(s)', 'NdZn2Al2(s)', 'GdZnMg12(s)', 'Nd5Mg41(s)',
    'Nd3Al11(s)', 'Nd3Zn22(s)', 'NdZn2Mg(s)', 'Nd2Zn9Mg5(s)',
    'NdZn2Al2(s)', 'Mg2Ca(s)', 'Al2Ca(s)', 'CaMgSi(s)', 'Mn2CaAl10(s)',
    'Ca2Mg55Zn43(s)', 'Y(s)', 'AlY(s)', 'Al3Y(s)', 'Al2Y3(s)',
    'Y6Mn23(s)', 'YZnMg12(s)', 'Mn(s)', 'Al4Mn(s)', 'Al11Mn4(s)',
    'Mn3Si(s)', 'Mn5Si3(s)', 'Mn2CaAl10(s)', 'Al7CuMn2(s)', 'Y6Mn23(s)',
    'Mn2Zr(s)', 'Zr(s)', 'Al3Zr(s)', 'Mn2Zr(s)', 'ZnZr(s)', 'Zn2Zr(s)',
    'Al30Mg23(s)', 'Al2Ca(s)', 'Mn2CaAl10(s)', 'Al7Cu3Mg6(s)',
    'Al5Cu6Mg2(s)', 'Al7CuMn2(s)', 'AlY(s)', 'Al3Y(s)', 'Al2Y3(s)',
    'Al3Zr(s)', 'Nd3Al11(s)', 'NdZn2Al2(s)', 'GdZnMg12(s)',
]

# Calculate sums for different element groups
Sum_column_Sr = df[['Sr2Mg17(s)', 'Sr5Si3(s)', 'SrZn5(s)', 'Sr2Zn43Mg55(s)']].sum(axis=1)
Sum_column_Zn = df[['Mg12Zn13(s)', 'Ca2Mg55Zn43(s)', 'SrZn5(s)', 'Sr2Zn43Mg55(s)',
                   'YZnMg12(s)', 'Zn2Zr(s)', 'ZnZr(s)', 'Nd3Zn22(s)', 'NdZn2Mg(s)',
                   'Nd2Zn9Mg5(s)', 'NdZn2Al2(s)']].sum(axis=1)
Sum_column_Nd = df[['Nd5Mg41(s)', 'Nd3Al11(s)', 'Nd3Zn22(s)', 'NdZn2Mg(s)',
                   'Nd2Zn9Mg5(s)', 'NdZn2Al2(s)']].sum(axis=1)
Sum_column_Ca = df[['Mg2Ca(s)', 'Al2Ca(s)', 'CaMgSi(s)', 'Mn2CaAl10(s)', 'Ca2Mg55Zn43(s)']].sum(axis=1)
Sum_column_Y = df[['Y(s)', 'AlY(s)', 'Al3Y(s)', 'Al2Y3(s)', 'Y6Mn23(s)', 'YZnMg12(s)']].sum(axis=1)
Sum_column_Mn = df[['Mn(s)', 'Al4Mn(s)', 'Al11Mn4(s)', 'Mn3Si(s)', 'Mn5Si3(s)',
                    'Mn2CaAl10(s)', 'Al7CuMn2(s)', 'Y6Mn23(s)', 'Mn2Zr(s)']].sum(axis=1)
Sum_column_Zr = df[['Zr(s)', 'Al3Zr(s)', 'Mn2Zr(s)', 'ZnZr(s)', 'Zn2Zr(s)']].sum(axis=1)
Sum_column_Al = df[['Al30Mg23(s)', 'Al2Ca(s)', 'Mn2CaAl10(s)', 'Al7Cu3Mg6(s)', 'Al5Cu6Mg2(s)',
                    'Al7CuMn2(s)', 'AlY(s)', 'Al3Y(s)', 'Al2Y3(s)', 'Al3Zr(s)',
                    'Nd3Al11(s)', 'NdZn2Al2(s)']].sum(axis=1)
Sum_column_Si = df[['Mg2Si(s)', 'CaMgSi(s)', 'Mn3Si(s)', 'Mn5Si3(s)', 'Sr5Si3(s)']].sum(axis=1)
Sum_column_Cu = df[['Mg2Cu(s)', 'Al7Cu3Mg6(s)', 'Al5Cu6Mg2(s)', 'Al7CuMn2(s)']].sum(axis=1)
Sum_column_Gd = df[['GdMg5(s)', 'GdZnMg12(s)']].sum(axis=1)

# Create a copy of the DataFrame and add the sum columns
df_copy = df.copy()
df_copy['Alkline'] = Sum_column_Sr + Sum_column_Ca
df_copy['Transition'] = Sum_column_Zr + Sum_column_Y + Sum_column_Mn + Sum_column_Zn + Sum_column_Cu
df_copy['Lanthanides'] = Sum_column_Nd + Sum_column_Gd
df_copy['Post-transition'] = Sum_column_Al
df_copy['Metalloids'] = Sum_column_Si
df_copy['SUM'] = df_copy[['Alkline', 'Transition', 'Lanthanides', 'Post-transition', 'Metalloids']].sum(axis=1)

# Drop the original element columns
df_drop = df_copy.drop(element_columns, axis=1)

# Encode the Heat Treatment column
lb = LabelBinarizer()
df_copy['Heat Treatment'] = lb.fit_transform(df_copy['Heat Treatment'])

# Define the target variables
target_UTS = df_copy.pop('UTS (MPa)')
target_YS = df_copy.pop('YS (MPa)')
target_EL = df_copy.pop('El%')

# Split the data into training and testing sets for UTS, YS, and EL
X_train_UTS, X_test_UTS, y_train_UTS, y_test_UTS = train_test_split(df_copy, target_UTS, test_size=0.2, random_state=27)
X_train_YS, X_test_YS, y_train_YS, y_test_YS = train_test_split(df_copy, target_YS, test_size=0.2, random_state=27)
X_train_EL, X_test_EL, y_train_EL, y_test_EL = train_test_split(df_copy, target_EL, test_size=0.2, random_state=27)

# Create a Random Forest Regressor for UTS, YS, and EL
rf_UTS = RandomForestRegressor(n_estimators=100, random_state=33)
rf_YS = RandomForestRegressor(n_estimators=100, random_state=33)
rf_EL = RandomForestRegressor(n_estimators=100, random_state=33)

# Fit the models for UTS, YS, and EL
rf_UTS.fit(X_train_UTS, y_train_UTS)
rf_YS.fit(X_train_YS, y_train_YS)
rf_EL.fit(X_train_EL, y_train_EL)

# Predictions for UTS, YS, and EL
y_pred_UTS = rf_UTS.predict(X_test_UTS)
y_pred_YS = rf_YS.predict(X_test_YS)
y_pred_EL = rf_EL.predict(X_test_EL)

# Evaluate the models for UTS, YS, and EL
def evaluate_model(y_true, y_pred, model_name):
    mae = metrics.mean_absolute_error(y_true, y_pred)
    rmse = metrics.mean_squared_error(y_true, y_pred, squared=False)
    r2 = metrics.r2_score(y_true, y_pred)
    print(f'{model_name} Model Evaluation:')
    print('MAE:', mae)
    print('RMSE:', rmse)
    print('R2 Score:', r2)

evaluate_model(y_test_UTS, y_pred_UTS, 'UTS')
evaluate_model(y_test_YS, y_pred_YS, 'YS')
evaluate_model(y_test_EL, y_pred_EL, 'EL')

# Set the style for plotting
sn.set_theme(style="ticks")
f, ax = plt.subplots(figsize=(11, 8))
sn.despine(f)

# Plot the histogram for YS with respect to Heat Treatment_T6
sn.histplot(df_copy, x="YS (MPa)", hue="Heat Treatment_T6", multiple="stack", edgecolor=".3", linewidth=.5, legend=False, bins=18)
plt.xlabel("YS (MPa)", fontsize=30)
plt.ylabel("Count", fontsize=30)
plt.rcParams.update({'font.size': 32})

# Visualize histograms for other features
features = df_copy.columns[3:26]
for feature in features:
    plt.rcParams.update({'font.size': 24})
    sn.histplot(data=df_copy[df_copy != 0], x=feature, bins=15, color="#53868B")
    plt.show()

# Create SHAP explainers and calculate SHAP values
explainer_UTS = shap.TreeExplainer(rf_UTS)
shap_values_UTS = explainer_UTS.shap_values(df_copy)

explainer_YS = shap.TreeExplainer(rf_YS)
shap_values_YS = explainer_YS.shap_values(df_copy)

# Visualize SHAP summary plots for UTS and YS
shap.summary_plot(shap_values_UTS, df_copy, plot_type="bar")
plt.xlabel("SHAP value (impact on model output)", fontsize=23)
plt.title('Feature Importance - UTS', fontsize=23)
plt.show()

shap.summary_plot(shap_values_YS, df_copy, plot_type="bar")
plt.xlabel("SHAP value (impact on model output)", fontsize=23)
plt.title('Feature Importance - YS', fontsize=23)
plt.show()

# Visualize SHAP beeswarm plots
shap.plots.beeswarm(shap_values_UTS, max_display=10, show=False)
plt.xlabel("SHAP value (impact on model output)", fontsize=23)
plt.title('Feature Importance - UTS', fontsize=23)
plt.show()

shap.plots.beeswarm(shap_values_YS, max_display=10, show=False)
plt.xlabel("SHAP value (impact on model output)", fontsize=23)
plt.title('Feature Importance - YS', fontsize=23)
plt.show()

# Visualize feature importance based on heat treatment status
heat_treated = ["Non heat treated" if shap_values_UTS[i, 11] == 0 else "Heat treated" for i in range(shap_values_UTS.shape[0])]
shap.plots.bar(shap_values_UTS.cohorts(heat_treated).abs.mean(0), max_display=10, show=False)
plt.xlabel("SHAP value (impact on model output)", fontsize=23)
plt.title('Feature Importance - UTS', fontsize=23)
plt.legend(fontsize=23)
plt.show()

# Calculate permutation importance for YS
perm_importance_YS = permutation_importance(rf_YS, X_test_YS, y_test_YS)
sorted_idx_YS = perm_importance_YS.importances_mean.argsort()
plt.figure(figsize=(10, 10))
plt.barh(features[sorted_idx_YS], perm_importance_YS.importances_mean[sorted_idx_YS])
plt.xlabel("Permutation Importance")
plt.title('Permutation Importance - YS')
plt.show()

# Continue to calculate permutation importance for UTS
perm_importance_UTS = permutation_importance(rf_UTS, X_test_UTS, y_test_UTS)
sorted_idx_UTS = perm_importance_UTS.importances_mean.argsort()
plt.figure(figsize=(10, 10))
plt.barh(features[sorted_idx_UTS], perm_importance_UTS.importances_mean[sorted_idx_UTS])
plt.xlabel("Permutation Importance")
plt.title('Permutation Importance - UTS')
plt.show()

# Define a list of models for further evaluation
models = [
    ('Random Forest (n=100)', RandomForestRegressor(n_estimators=100)),
    ('SVR (Linear)', SVR(kernel='linear')),
    ('SVR (poly)', SVR(kernel='poly')),
    ('SVR (rbf)', SVR(kernel='rbf')),
    ('Linear regression', LinearRegression()),
    ('Lasso', linear_model.Lasso(alpha=0.1))
]

# Define a list of target variable names
target_names = ['UTS', 'YS', 'EL']

# Iterate through target variables and models to evaluate performance
for target_name in target_names:
    print(f"---------------------- {target_name} ----------------------")
    X_target = df_copy.drop(target_name, axis=1)
    Y_target = df_copy[target_name]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_target, Y_target, test_size=0.15, shuffle=True, random_state=2)
    
    for model_name, model_instance in models:
        model = model_instance
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate and print evaluation metrics
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        r2 = metrics.r2_score(y_test, y_pred)
        
        print(f"Model: {model_name}")
        print("--------------------------------------")
        print('MAE:', mae)
        print('MSE:', mse)
        print('R2 Score:', r2)

# Calculate Spearman correlation matrix
correlation_matrix = df_copy.corr(method="spearman")

# Plot the Spearman correlation matrix
plt.figure(figsize=(33, 3))
plt.rcParams.update({'font.size': 24, 'font.weight': 'bold'})
heatmap = sn.heatmap(correlation_matrix.iloc[:2, 3:], vmin=-1, vmax=1, annot=True, cmap="RdYlGn")
plt.title("Spearman Correlation")
plt.show()

# Create a mask for upper triangle of the correlation matrix
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Plot the Spearman correlation matrix with the upper triangle masked
plt.figure(figsize=(50, 50))
plt.rcParams.update({'font.size': 60, 'font.weight': 'bold'})
heatmap = sn.heatmap(correlation_matrix, mask=mask, vmin=-1, vmax=1, annot=False, cmap="RdYlGn")
plt.show()

# Create and fit a Random Forest Regressor for UTS
rf_UTS = RandomForestRegressor(n_estimators=14, random_state=27)
rf_UTS.fit(X_target, Y_target)

# Define a list of features for Partial Dependence Plots (PDP)
features = df_copy.columns[3:21]

# Generate Partial Dependence Plots for selected feature pairs
for i in range(15):
    for j in range(i + 1, 15):
        feature_pair = [(i, j)]
        my_plot = plot_partial_dependence(rf_UTS, features=feature_pair, X=X_target, feature_names=features, grid_resolution=50)
        plt.subplots_adjust(wspace=0.2, hspace=0.5)
        plt.show()

# Create SHAP explainers for UTS, YS, and EL
explainer_UTS = shap.TreeExplainer(rf_UTS)
explainer_YS = shap.TreeExplainer(rf_YS)
explainer_EL = shap.TreeExplainer(rf_EL)

# Calculate SHAP values for UTS, YS, and EL
shap_values_UTS = explainer_UTS.shap_values(X_target)
shap_values_YS = explainer_YS.shap_values(X_target)
shap_values_EL = explainer_EL.shap_values(X_target)

# Create SHAP summary plots for UTS, YS, and EL
shap.summary_plot(shap_values_UTS, X_target, plot_type="bar")
plt.xlabel("SHAP value (impact on model output)", fontsize=23)
plt.title('Feature Importance - UTS', fontsize=23)
plt.show()

shap.summary_plot(shap_values_YS, X_target, plot_type="bar")
plt.xlabel("SHAP value (impact on model output)", fontsize=23)
plt.title('Feature Importance - YS', fontsize=23)
plt.show()

shap.summary_plot(shap_values_EL, X_target, plot_type="bar")
plt.xlabel("SHAP value (impact on model output)", fontsize=23)
plt.title('Feature Importance - EL', fontsize=23)
plt.show()
