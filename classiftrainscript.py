import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

df = pd.read_csv(r"C:\Users\admin\Downloads\Milestone 2 Updated Datasets\Milestone 2 Updated Datasets\Datasets train splits\Vitamin Deficiency Predcition\train_data.csv")
df.head()

#understanding the dataset
df.shape
df.head()
df.info()
df.describe()

#total null value
df.isnull().sum()
#handling null values
#creates flag, 1 if missing, 0 if not missing
df['alcohol_missing'] = df['alcohol_consumption'].isna().astype(int)
df['symptoms_missing'] = df['symptoms_list'].isna().astype(int)

df['alcohol_consumption'] = df['alcohol_consumption'].fillna('Unknown')
df['symptoms_list'] = df['symptoms_list'].fillna('None')


#no duplicates found
duplicates_total = df.duplicated().sum()
print(duplicates_total)

df = df.drop(columns=['symptoms_list'])


# Get the value counts for the 'disease_diagnosis' column
disease_counts = df['disease_diagnosis'].value_counts()

print("Distribution of disease_diagnosis:")
print(disease_counts)
print("\n")

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Subplot 1: Count Plot
sns.countplot(data=df, y='disease_diagnosis', order=disease_counts.index, palette='viridis', ax=axes[0], legend=False)
axes[0].set_title('Distribution of Disease Diagnosis (Count Plot)')
axes[0].set_xlabel('Count')
axes[0].set_ylabel('Disease Diagnosis')

# Subplot 2: Pie Chart
axes[1].pie(disease_counts, labels=disease_counts.index, autopct='%1.1f%%', startangle=90)
axes[1].set_title('Proportion of Disease Diagnosis (Pie Chart)')
axes[1].set_ylabel('') # Hide the default 'disease_diagnosis' label on y-axis for pie chart

plt.tight_layout()
plt.show()

#feature engineering

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

df['total_vitamin_intake'] = df[vitamin_cols].sum(axis=1)
df['vitamin_gap'] = (100 - df[vitamin_cols]).clip(lower=0).sum(axis=1)


#feature encoding
#handling target variable
from sklearn.preprocessing import OneHotEncoder

y = df['disease_diagnosis']
X = df.drop(columns=['disease_diagnosis'])

cat_features = [
    'gender','smoking_status','alcohol_consumption',
    'exercise_level','diet_type','sun_exposure',
    'income_level','latitude_region'
]

encoder = OneHotEncoder(
    drop='first',
    handle_unknown='ignore',
    sparse_output=False
)

X_cat_encoded = encoder.fit_transform(X[cat_features])

X_cat_encoded_df = pd.DataFrame(
    X_cat_encoded,
    columns=encoder.get_feature_names_out(cat_features),
    index=X.index
)


X_encoded = pd.concat(
    [X.drop(columns=cat_features), X_cat_encoded_df],
    axis=1
)

# ==============================
# ENCODE TARGET
# ==============================
y_encoded, class_names = pd.factorize(y)


#feature selection using random forest importance selector
rf_selector = RandomForestClassifier(n_estimators=200, random_state=42)
rf_selector.fit(X_encoded, y_encoded)

importances = pd.Series(rf_selector.feature_importances_, index=X_encoded.columns)

top_k = 12
top_features = importances.sort_values(ascending=False).head(top_k).index.tolist()

X_selected = X_encoded[top_features]

print("Selected Features:")
print(top_features)

X_selected.corrwith(pd.Series(y_encoded, index=X_selected.index)).sort_values(ascending=False)

#split and scale

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y_encoded, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_results = []

C_values = [0.01, 1, 10]
max_iter_values = [100, 300, 500]

# Vary C (fix max_iter)
for C in C_values:
    model = LogisticRegression(C=C, max_iter=300, class_weight = 'balanced')

    start_train = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_train

    start_test = time.time()
    preds = model.predict(X_test_scaled)
    test_time = time.time() - start_test

    acc = accuracy_score(y_test, preds)

    lr_results.append(("C="+str(C), acc, train_time, test_time))

# Vary max_iter (fix C)
for m in max_iter_values:
    model = LogisticRegression(C=10, max_iter=m,class_weight = 'balanced')

    start_train = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_train

    start_test = time.time()
    preds = model.predict(X_test_scaled)
    test_time = time.time() - start_test

    acc = accuracy_score(y_test, preds)

    lr_results.append(("iter="+str(m), acc, train_time, test_time))

print("Logistic Regression Results:")
for result in lr_results:
    print(f"  Parameters: {result[0]}, Accuracy: {result[1]:.4f}, Train Time: {result[2]:.4f}s, Test Time: {result[3]:.4f}s")

rf_results = []

n_values = [50, 100, 200]
depth_values = [5, 10, 20]

# Vary n_estimators
for n in n_values:
    model = RandomForestClassifier(n_estimators=n, max_depth=10, random_state=42,class_weight = 'balanced')

    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train

    start_test = time.time()
    preds = model.predict(X_test)
    test_time = time.time() - start_test

    acc = accuracy_score(y_test, preds)

    rf_results.append(("n="+str(n), acc, train_time, test_time))

# Vary max_depth
for d in depth_values:
    model = RandomForestClassifier(n_estimators=100, max_depth=d, random_state=42,class_weight = 'balanced')

    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train

    start_test = time.time()
    preds = model.predict(X_test)
    test_time = time.time() - start_test

    acc = accuracy_score(y_test, preds)

    rf_results.append(("depth="+str(d), acc, train_time, test_time))

print("Random Forest Results:")
for result in rf_results:
    print(f"  Parameters: {result[0]}, Accuracy: {result[1]:.4f}, Train Time: {result[2]:.4f}s, Test Time: {result[3]:.4f}s")

knn_results = []

k_values = [3, 5, 7]
weight_values = ['uniform', 'distance']

# Vary k
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k, weights='uniform')

    start_train = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_train

    start_test = time.time()
    preds = model.predict(X_test_scaled)
    test_time = time.time() - start_test

    acc = accuracy_score(y_test, preds)

    knn_results.append(("k="+str(k), acc, train_time, test_time))

# Vary weights
for w in weight_values:
    model = KNeighborsClassifier(n_neighbors=5, weights=w)

    start_train = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_train

    start_test = time.time()
    preds = model.predict(X_test_scaled)
    test_time = time.time() - start_test

    acc = accuracy_score(y_test, preds)

    knn_results.append(("w="+w, acc, train_time, test_time))

print("K-Nearest Neighbors Results:")
for result in knn_results:
    print(f"  Parameters: {result[0]}, Accuracy: {result[1]:.4f}, Train Time: {result[2]:.4f}s, Test Time: {result[3]:.4f}s")

svm_results = []

C_values = [0.1, 1, 10]
kernel_values = ['linear', 'rbf','poly']

# Vary C (fix kernel)
for C in C_values:
    model = SVC(C=C, kernel='rbf',class_weight = 'balanced')

    start_train = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_train

    start_test = time.time()
    preds = model.predict(X_test_scaled)
    test_time = time.time() - start_test

    acc = accuracy_score(y_test, preds)

    svm_results.append((f"C={C}", acc, train_time, test_time))

# Vary kernel (fix C)
for k in kernel_values:
    model = SVC(C=10, kernel=k,class_weight = 'balanced')

    start_train = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_train

    start_test = time.time()
    preds = model.predict(X_test_scaled)
    test_time = time.time() - start_test

    acc = accuracy_score(y_test, preds)

    svm_results.append((f"kernel={k}", acc, train_time, test_time))

print("Support Vector Machine Results:")
for result in svm_results:
    print(f"  Parameters: {result[0]}, Accuracy: {result[1]:.4f}, Train Time: {result[2]:.4f}s, Test Time: {result[3]:.4f}s")

best_lr = max(lr_results, key=lambda x: x[1])
best_rf = max(rf_results, key=lambda x: x[1])
best_knn = max(knn_results, key=lambda x: x[1])
best_svm = max(svm_results, key=lambda x: x[1])

models = ["Logistic", "Random Forest", "KNN", "SVM"]
accuracies = [best_lr[1], best_rf[1], best_knn[1], best_svm[1]]
train_times = [best_lr[2], best_rf[2], best_knn[2], best_svm[2]]
test_times = [best_lr[3], best_rf[3], best_knn[3], best_svm[3]]

print(f"Best Logistic Regression: Parameters: {best_lr[0]}, Accuracy: {best_lr[1]:.4f}, Train Time: {best_lr[2]:.4f}s, Test Time: {best_lr[3]:.4f}s")
print(f"Best Random Forest: Parameters: {best_rf[0]}, Accuracy: {best_rf[1]:.4f}, Train Time: {best_rf[2]:.4f}s, Test Time: {best_rf[3]:.4f}s")
print(f"Best K-Nearest Neighbors: Parameters: {best_knn[0]}, Accuracy: {best_knn[1]:.4f}, Train Time: {best_knn[2]:.4f}s, Test Time: {best_knn[3]:.4f}s")
print(f"Best SVM: Parameters: {best_svm[0]}, Accuracy: {best_svm[1]:.4f}, Train Time: {best_svm[2]:.4f}s, Test Time: {best_svm[3]:.4f}s")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Accuracy
axes[0].bar(models, accuracies)
axes[0].set_title("Model Accuracy")
axes[0].set_ylabel("Accuracy")

# Training Time
axes[1].bar(models, train_times)
axes[1].set_title("Training Time")
axes[1].set_ylabel("Time (s)")

# Testing Time
axes[2].bar(models, test_times)
axes[2].set_title("Testing Time")
axes[2].set_ylabel("Time (s)")

plt.tight_layout()
plt.show()

import pickle

# Save selected features
with open("top_features_cls.pkl", "wb") as f:
    pickle.dump(top_features, f)

# Save scaler
with open("scaler_cls.pkl", "wb") as f:
    pickle.dump(scaler, f)
# save encoder
with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

# Save label mapping
with open("label_mapping.pkl", "wb") as f:
    pickle.dump(class_names, f)

# Save best models only
best_lr_model = LogisticRegression(C=10, max_iter=300)
best_lr_model.fit(X_train_scaled, y_train)

best_rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
best_rf_model.fit(X_train, y_train)

best_knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance')
best_knn_model.fit(X_train_scaled, y_train)

best_svm_model = SVC(C=10, kernel='rbf')
best_svm_model.fit(X_train_scaled, y_train)

pickle.dump(best_lr_model, open("lr_cls.pkl", "wb"))
pickle.dump(best_rf_model, open("rf_cls.pkl", "wb"))
pickle.dump(best_knn_model, open("knn_cls.pkl", "wb"))
pickle.dump(best_svm_model, open("svm_cls.pkl", "wb"))

print("Classification models saved")

