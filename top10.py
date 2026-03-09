#Import semua library yang diperlukan
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight

#Memanggil API kaggle untuk mendownload dataset

#path = kagglehub.dataset_download("uom190346a/disease-symptoms-and-patient-profile-dataset")
#print("Dataset folder:", path)
#print(os.listdir(path))

#SKIP-suka error pas dibuka di device lain

#Load dataset
url = "https://raw.githubusercontent.com/flywzen/ANN-IB/25d228ee85b87bb9018948b01c6de85c309aa9a6/Disease_symptom_and_patient_profile_dataset.csv"
df = pd.read_csv(url)

#Buat load dataset lokal

#csv_name = [f for f in os.listdir(path) if f.endswith(".csv")][0]
#df = pd.read_csv(os.path.join(path, csv_name))

#Interactive sheet untuk dataset
#sheet = sheets.InteractiveSheet(df=df)

import matplotlib.pyplot as plt
import seaborn as sns

disease = df["Disease"].value_counts().head(116)

plt.figure(figsize=(64,32))
sns.barplot(x=disease.index, y=disease.values)
plt.xticks(rotation=45)
plt.title("Distribusi Penyakit")
plt.ylabel("Jumlah")
plt.xlabel("Disease")
plt.show()

print("\nPenyakit Terbanyak:")
print(disease)


top10 = df["Disease"].value_counts().head(10)

plt.figure(figsize=(8,4))
sns.barplot(x=top10.index, y=top10.values)
plt.xticks(rotation=45)
plt.title("Top 10 Penyakit Terbanyak")
plt.ylabel("Jumlah")
plt.xlabel("Disease")
plt.show()

print("\nTop 10 Penyakit:")
print(top10)


# Ambil top 10 diseases
top3 = ["Asthma", "Stroke", "Osteoporosis", "Diabetes", "Hypertension", "Migraine", "Influenza", "Bronchitis", "Pneumonia", "Hyperthyroidism"]
df = df[df["Disease"].isin(top3)].reset_index(drop=True)

print("Jumlah data:", df.shape)

#Cek missing value
X = df.drop(columns=["Disease"])
y = df["Disease"]

#Pisahkan fitur & label
X = df.drop(columns=["Disease"])
y = df["Disease"]


#Encode label penyakit
label_map = {label: idx for idx, label in enumerate(sorted(y.unique()))}
y_encoded = y.map(label_map)

print("Label map:", label_map)

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

#Preprocessing Pipeline
numeric_features = ["Age"]
categorical_features = [col for col in X.columns if col != "Age"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

X_train = preprocess.fit_transform(X_train)
X_test = preprocess.transform(X_test)

X_train = X_train.toarray() if hasattr(X_train, "toarray") else X_train
X_test = X_test.toarray() if hasattr(X_test, "toarray") else X_test

num_classes = len(np.unique(y_encoded))

#Class Weights
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_encoded),
    y=y_encoded
)

class_weights = {i: class_weights[i] for i in range(len(class_weights))}
print("Class Weights:", class_weights)

#Build model
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(16, activation="relu"),
    Dropout(0.3),
    Dense(8, activation="relu"),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

#Training
es = EarlyStopping(
    patience=15,
    restore_best_weights=True,
    monitor="val_loss"
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=150,
    batch_size=4,
    class_weight=class_weights,
    callbacks=[es],
    verbose=1
)

#Evaluasi
pred = np.argmax(model.predict(X_test), axis=1)

acc = accuracy_score(y_test, pred)
print("\nAccuracy:", acc)

print("\nClassification Report:")
print(classification_report(y_test, pred, target_names=label_map.keys()))

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)


plt.figure(figsize=(8,5))
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.axhline(test_acc, color="red", linestyle="--", label="Test Acc")
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.axhline(test_loss, color="red", linestyle="--", label="Test Loss")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()


cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_map.keys(),
            yticklabels=label_map.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix on Test Set")
plt.show()


from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import numpy as np


# 1. PREPROCESSING SEKALI



X_all = preprocess.fit_transform(X)
X_all = X_all.toarray() if hasattr(X_all, "toarray") else X_all
y_all = y_encoded.to_numpy()


# 2. SETUP K-FOLD


k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

fold_accuracies = []

def build_model():
    model = Sequential([
        Input(shape=(X_all.shape[1],)),
        Dense(16, activation="relu"),
        Dropout(0.3),
        Dense(8, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# 3. LOOP TRAINING PER FOLD


for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
    print(f"\n🔵 FOLD {fold+1}/{k}")

    X_tr, X_val = X_all[train_idx], X_all[val_idx]
    y_tr, y_val = y_all[train_idx], y_all[val_idx]

    model = build_model()

    es = EarlyStopping(
        patience=7,
        restore_best_weights=True,
        monitor="val_loss"
    )

    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=40,
        batch_size=4,
        callbacks=[es],
        verbose=0
    )

    pred = np.argmax(model.predict(X_val), axis=1)
    acc = accuracy_score(y_val, pred)
    fold_accuracies.append(acc)

    print(f"Fold {fold+1} Accuracy: {acc:.4f}")


# 4. RATA-RATA HASIL

print("\n===========================")
print("K-Fold Evaluation Summary")
print("===========================")
print("Akurasi per fold:", fold_accuracies)
print("Rata-rata akurasi:", np.mean(fold_accuracies))
print("Standar deviasi:", np.std(fold_accuracies))