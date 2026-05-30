import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import accuracy_score, classification_report

# LOAD TEST DATA

df_test = pd.read_csv(r"C:\Users\admin\Downloads\test_data_cls.csv")
class_names = pickle.load(open("label_mapping.pkl", "rb"))

y_test = pd.Categorical( df_test['disease_diagnosis'],categories=class_names).codes
X_test = df_test.drop(columns=['disease_diagnosis'])


# LOAD ARTIFACTS

top_features = pickle.load(open("top_features_cls.pkl", "rb"))
scaler = pickle.load(open("scaler_cls.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
class_names = pickle.load(open("label_mapping.pkl", "rb"))

lr_model = pickle.load(open("lr_cls.pkl", "rb"))
rf_model = pickle.load(open("rf_cls.pkl", "rb"))
knn_model = pickle.load(open("knn_cls.pkl", "rb"))
svm_model = pickle.load(open("svm_cls.pkl", "rb"))



# MISSING VALUES HANDLING

X_test['alcohol_missing'] = X_test['alcohol_consumption'].isna().astype(int)
X_test['symptoms_missing'] = X_test['symptoms_list'].isna().astype(int)

X_test['alcohol_consumption'] = X_test['alcohol_consumption'].fillna('Unknown')
X_test['symptoms_list'] = X_test['symptoms_list'].fillna('None')

X_test = X_test.drop(columns=['symptoms_list'])


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


cat_features = [
    'gender','smoking_status','alcohol_consumption',
    'exercise_level','diet_type','sun_exposure',
    'income_level','latitude_region'
]


# ONE HOT ENCODING

X_cat = encoder.transform(X_test[cat_features])

X_cat_df = pd.DataFrame(
    X_cat,
    columns=encoder.get_feature_names_out(cat_features),
    index=X_test.index
)


X_test_encoded = pd.concat(
    [X_test.drop(columns=cat_features), X_cat_df],
    axis=1
)


# ALIGN FEATURES
X_test_encoded = X_test_encoded.reindex(columns=top_features, fill_value=0)


# SCALING
X_test_scaled = scaler.transform(X_test_encoded)


# PREDICTIONS

pred_lr = lr_model.predict(X_test_scaled)
pred_rf = rf_model.predict(X_test_encoded)
pred_knn = knn_model.predict(X_test_scaled)
pred_svm = svm_model.predict(X_test_scaled)


# EVALUATION

print("\n=== LOGISTIC REGRESSION ===")
print("Accuracy:", accuracy_score(y_test, pred_lr))
print(classification_report(y_test, pred_lr, target_names=class_names))

print("\n=== RANDOM FOREST ===")
print("Accuracy:", accuracy_score(y_test, pred_rf))
print(classification_report(y_test, pred_rf, target_names=class_names))

print("\n=== KNN ===")
print("Accuracy:", accuracy_score(y_test, pred_knn))
print(classification_report(y_test, pred_knn, target_names=class_names))

print("\n=== SVM ===")
print("Accuracy:", accuracy_score(y_test, pred_svm))
print(classification_report(y_test, pred_svm, target_names=class_names))