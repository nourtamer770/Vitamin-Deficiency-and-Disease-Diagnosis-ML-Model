
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import mean_squared_error, r2_score


# LOAD TEST DATA

df_test = pd.read_csv(r"C:\Users\admin\Downloads\test_data_reg.csv")

y_test = df_test['vitamin_deficiency']
X_test = df_test.drop(columns=['vitamin_deficiency'], errors='ignore')


# LOAD SAVED OBJECTS

encoder = pickle.load(open("encoder.pkl", "rb"))
trained_columns = pickle.load(open("encoded_columns.pkl", "rb"))

high_corr_features = pickle.load(open("high_corr_features.pkl", "rb"))
top_features = pickle.load(open("top_features.pkl", "rb"))

scaler_high_corr = pickle.load(open("scaler_high_corr.pkl", "rb"))
scaler_fs = pickle.load(open("scaler_feature_selection.pkl", "rb"))

# Models
lr_high_corr = pickle.load(open("lr_high_corr.pkl", "rb"))
rf_high_corr = pickle.load(open("rf_high_corr.pkl", "rb"))
cat_high_corr = pickle.load(open("cat_high_corr.pkl", "rb"))

lr_fs = pickle.load(open("lr_fs.pkl", "rb"))
rf_fs = pickle.load(open("rf_fs.pkl", "rb"))
cat_fs = pickle.load(open("cat_fs.pkl", "rb"))

# MISSING VALUES HANDLING

X_test['alcohol_missing'] = X_test['alcohol_consumption'].isna().astype(int)
X_test['symptoms_missing'] = X_test['symptoms_list'].isna().astype(int)

X_test['alcohol_consumption'] = X_test['alcohol_consumption'].fillna('Unknown')
X_test['symptoms_list'] = X_test['symptoms_list'].fillna('None')


# FEATURE ENGINEERING

vitamin_cols = [
    'vitamin_a_percent_rda',
    'vitamin_c_percent_rda',
    'vitamin_d_percent_rda',
    'vitamin_e_percent_rda',
    'vitamin_b12_percent_rda',
    'folate_percent_rda',
    'calcium_percent_rda',
    'iron_percent_rda'
]

X_test['total_vitamin_intake'] = X_test[vitamin_cols].sum(axis=1)
X_test['vitamin_gap'] = (100 - X_test[vitamin_cols]).clip(lower=0).sum(axis=1)

X_test = X_test.drop(columns=['symptoms_list'], errors='ignore')


# ONE HOT ENCODING

cat_features = [
    'gender','smoking_status','alcohol_consumption',
    'exercise_level','diet_type','sun_exposure',
    'income_level','latitude_region'
]

encoded_array = encoder.transform(X_test[cat_features])

encoded_df = pd.DataFrame(
    encoded_array,
    columns=encoder.get_feature_names_out(cat_features),
    index=X_test.index
)

X_test_encoded = pd.concat(
    [X_test.drop(columns=cat_features), encoded_df],
    axis=1
)


# ALIGN WITH TRAINING

X_test_encoded = X_test_encoded.reindex(columns=trained_columns, fill_value=0)


# PIPELINE 1: HIGH CORR

X_high = X_test_encoded[high_corr_features]
X_high_scaled = scaler_high_corr.transform(X_high)

pred_lr_high = lr_high_corr.predict(X_high_scaled)
pred_rf_high = rf_high_corr.predict(X_high)
pred_cat_high = cat_high_corr.predict(X_high)


# PIPELINE 2: FEATURE SELECTION

X_fs = X_test_encoded[top_features]
X_fs_scaled = scaler_fs.transform(X_fs)

pred_lr_fs = lr_fs.predict(X_fs_scaled)
pred_rf_fs = rf_fs.predict(X_fs)
pred_cat_fs = cat_fs.predict(X_fs)


# EVALUATION (ONLY BECAUSE LABELS EXIST)
def evaluate(y_true, y_pred, name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name} -> MSE: {mse:.4f}, R2: {r2:.4f}")

print("\n--- HIGH CORRELATION PIPELINE ---")
evaluate(y_test, pred_lr_high, "LR")
evaluate(y_test, pred_rf_high, "RF")
evaluate(y_test, pred_cat_high, "CatBoost")

print("\n--- FEATURE SELECTION PIPELINE ---")
evaluate(y_test, pred_lr_fs, "LR")
evaluate(y_test, pred_rf_fs, "RF")
evaluate(y_test, pred_cat_fs, "CatBoost")