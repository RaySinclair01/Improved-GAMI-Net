# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
# Load the Excel file
excel_path = 'e50_df_train_test_set.xlsx'
train_sheet_name = 'Train_Set'
test_sheet_name = 'Test_Set'

# Read the training and testing sheets
df_train = pd.read_excel(excel_path, sheet_name=train_sheet_name)
df_test = pd.read_excel(excel_path, sheet_name=test_sheet_name)

# Data Preprocessing
X_train = df_train.drop(columns=['site', 'Chl'])
y_train = df_train['Chl']
X_test = df_test.drop(columns=['site', 'Chl'])
y_test = df_test['Chl']

# Initialize models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
svm_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
ann_model = MLPRegressor(hidden_layer_sizes=(24, 24), max_iter=500, random_state=42)

# Metrics storage
model_metrics = {'GAMINet':{},'RF': {}, 'SVM': {}, 'ANN': {}}
# Sample GAMI-Net metrics (You would replace these with your actual calculated values)
gaminet_metrics = {
    'R2_Train': 0.98,  # Replace with your actual value
    'RMSE_Train': 0.03,  # Replace with your actual value
    'RPD_Train': 7.35,  # Replace with your actual value
    'R2_Test': 0.90,  # Replace with your actual value
    'RMSE_Test': 0.09,  # Replace with your actual value
    'RPD_Test': 3.2  # Replace with your actual value
}

# Add GAMI-Net metrics into model_metrics dictionary
model_metrics['GAMINet'] = gaminet_metrics

# Train and evaluate models
for model_name, model in zip(['RF', 'SVM', 'ANN'], [rf_model, svm_model, ann_model]):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rpd_train = np.std(y_train) / rmse_train

    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    rpd_test = np.std(y_test) / rmse_test

    model_metrics[model_name]['R2_Train'] = r2_train
    model_metrics[model_name]['RMSE_Train'] = rmse_train
    model_metrics[model_name]['RPD_Train'] = rpd_train
    model_metrics[model_name]['R2_Test'] = r2_test
    model_metrics[model_name]['RMSE_Test'] = rmse_test
    model_metrics[model_name]['RPD_Test'] = rpd_test

# Retrain the ANN model for loss curve
ann_model_verbose = MLPRegressor(hidden_layer_sizes=(24, 24), max_iter=500, random_state=42)
ann_model_verbose.fit(X_train, y_train)
loss_curve = ann_model_verbose.loss_curve_

# Parameters for convergence curves
n_iterations = 20  # Number of iterations for averaging
n_trees_range = np.arange(10, 210, 10)  # Number of trees for RF
c_values = np.logspace(-2, 2, 10)  # C values for SVM

# Storage for metrics
rf_rmse_avg = []
rf_r2_avg = []
svm_rmse_avg = []
svm_r2_avg = []

# RF Convergence Curve
for n_trees in n_trees_range:
    rmse_values = []
    r2_values = []
    for _ in range(n_iterations):
        rf = RandomForestRegressor(n_estimators=n_trees)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_train)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        r2 = r2_score(y_train, y_pred)
        rmse_values.append(rmse)
        r2_values.append(r2)
    rf_rmse_avg.append(np.mean(rmse_values))
    rf_r2_avg.append(np.mean(r2_values))

# SVM Convergence Curve
for c in c_values:
    rmse_values = []
    r2_values = []
    for _ in range(n_iterations):
        svm = SVR(C=c)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_train)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        r2 = r2_score(y_train, y_pred)
        rmse_values.append(rmse)
        r2_values.append(r2)
    svm_rmse_avg.append(np.mean(rmse_values))
    svm_r2_avg.append(np.mean(r2_values))
# In[]:
# Plotting
# Create a 2x1 grid for the main layout
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5])

# Create a 3x1 grid for the convergence curves on the left side
gs_left = gs[0].subgridspec(3, 1, hspace=0.3)  # 添加 hspace 参数)
left_axes = [fig.add_subplot(gs_left[i, 0]) for i in range(3)]

# Create a 2x3 grid for the performance metrics on the right side
gs_right = gs[1].subgridspec(2, 3,wspace=0.3)   # 2 rows, 3 columns for Train/Test and R2/RMSE/RPD
right_axes = [[fig.add_subplot(gs_right[i, j]) for j in range(3)] for i in range(2)]



# Convergence Curve for ANN
left_axes[0].plot(loss_curve, label='ANN Loss Curve', color="#E45C5E")
left_axes[0].set_xlabel('Iterations')
left_axes[0].set_ylabel('Loss')
left_axes[0].set_title('Convergence Curve (ANN)')

# Convergence Curve for RF
left_axes[1].plot(n_trees_range, rf_rmse_avg, label='RF RMSE', color="#004775")
#left_axes[1].plot(n_trees_range, rf_r2_avg, label='RF R2', color="#004775")
left_axes[1].set_xlabel('Number of Trees')
left_axes[1].set_ylabel('RMSE')
left_axes[1].set_title('Convergence Curve (RF)')

# Convergence Curve for SVM
left_axes[2].semilogx(c_values, svm_rmse_avg, label='SVM RMSE', color="#41A0BC")
#left_axes[2].semilogx(c_values, svm_r2_avg, label='SVM R2', color="#41A0BC")
left_axes[2].set_xlabel('C Values')
left_axes[2].set_ylabel('RMSE')
left_axes[2].set_title('Convergence Curve (SVM)')

# Add legends to left subplots
for ax in left_axes:
    ax.legend()
# For making the left subplot borders thicker
for ax in left_axes:
    for spine in ax.spines.values():
        spine.set_linewidth(2)
# ... (Code for plotting performance metrics on the right side, using right_axes)
# In[]:
# Plotting performance metrics on the right side
new_colors = {'GAMINet': '#004775', 'RF': '#41A0BC', 'SVM': '#E45C5E', 'ANN': '#F7A947'}

metrics = ['R2', 'RMSE', 'RPD']
# Loop for train and test sets
for i, data_type in enumerate(['Train', 'Test']):
    for j, metric in enumerate(metrics):
        ax = right_axes[i][j]
        metric_data = f"{metric}_{data_type}"
        metric_values = [model_metrics[model][metric_data] for model in ['GAMINet', 'RF', 'SVM', 'ANN']]

        for k, model in enumerate(['GAMINet', 'RF', 'SVM', 'ANN']):
            ax.bar(k, metric_values[k], color=new_colors[model], width=0.8,
                   label=model if i == 0 and j == 0 else "")

        ax.set_xticks(range(len(['GAMINet', 'RF', 'SVM', 'ANN'])))
        ax.set_xticklabels(['GAMINet', 'RF', 'SVM', 'ANN'])
        ax.set_title(f"{metric} ({data_type} Set)")

# For making the right subplot borders thicker
for row_axes in right_axes:
    for ax in row_axes:
        for spine in ax.spines.values():
            spine.set_linewidth(2)

# Add a single legend at the top
fig.legend(['GAMINet', 'RF', 'SVM', 'ANN'], title='Models', loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=4)


# Add a single legend at the top
handles, labels = right_axes[0][0].get_legend_handles_labels()  # Get handles and labels from one of the subplots
fig.legend(handles, labels, title='Models', loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=4)

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make space for the legend
plt.savefig('GAMINet_RF_SVM_ANN.png', dpi=600)
plt.show()

# In[]:
# Save model metrics to Excel
metrics_df = pd.DataFrame(model_metrics)
metrics_df.index = pd.MultiIndex.from_product([['Train', 'Test'], ['R2', 'RMSE', 'RPD']], names=['Set', 'Metric'])

# Save to Excel
excel_save_path = 'GAMINet_RF_SVM_ANN_metrics.xlsx'
with pd.ExcelWriter(excel_save_path) as writer:
    metrics_df.to_excel(writer, sheet_name='Model Metrics')
