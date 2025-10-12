import os
from datetime import datetime
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from fastdtw import fastdtw
from rich.console import Console
from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

console = Console()
console.print("\n[bold cyan]>>> STARTING DTW + k-NN MODEL <<<\n")


console.print("[bold]Loading normalized data...[/bold]")

NORMALIZED_DATA_PATH = "data/processed_features/01_normalized_sequences_pos.pkl"
try:
    with open(NORMALIZED_DATA_PATH, "rb") as f:
        data = pickle.load(f)

    sequences_raw = data["sequences"]
    labels_raw = data["labels"]
    console.print(
        f"[green]Successfully loaded {len(sequences_raw)} raw sequences.[/green]"
    )
except FileNotFoundError:
    console.print(
        f"[bold red] ERROR: File '{NORMALIZED_DATA_PATH}' not found. Run the normalization script first.[/bold red]"
    )
    exit()

MIN_SEQUENCE_LENGTH = 2
sequences = []
labels = []
for seq_list, label in zip(sequences_raw, labels_raw):
    if len(seq_list) >= MIN_SEQUENCE_LENGTH:
        sequences.append(np.array(seq_list, dtype=np.float64))

        labels.append(label.split(" ")[0])
        # ? Optional: not aggregate labels to 5 main classes
        # labels.append(label)

console.print(f"Filtered ended to [bold]{len(sequences)}[/bold] valid sequences.")


console.print("\n[bold]Preparing labels and splitting data...[/bold]")

# Changing text labels to integer representation
label_encoder = LabelEncoder()
y_numeric = label_encoder.fit_transform(labels)

console.print("[underline]Mapped labels to numbers:[/underline]")
for i, class_name in enumerate(label_encoder.classes_):
    console.print(f"  [magenta]{i}[/magenta]: {class_name}")

# Instead of splitting the data itself, create a list of indices [0, 1, 2, ...] and split it
indices = np.arange(len(sequences))

# `stratify=y_numeric` ensures that the class proportions are the same in the training and test sets
X_train_indices, X_test_indices, y_train, y_test = train_test_split(
    indices, y_numeric, test_size=0.2, random_state=42, stratify=y_numeric
)

# Create the actual datasets based on the split indices
X_train = [sequences[i] for i in X_train_indices]
X_test = [sequences[i] for i in X_test_indices]

console.print(f"Training set size: [bold]{len(X_train)}[/bold] samples")
console.print(f"Test set size: [bold]{len(X_test)}[/bold] samples")


console.print("\n[bold]Calculating DTW distance matrix...[/bold]")


def dtw_distance(s1, s2):
    """Calculates the DTW distance between two sequences."""
    distance, _ = fastdtw(s1, s2, dist=cosine)
    return distance


n_test = len(X_test)
n_train = len(X_train)

distance_matrix = np.zeros((n_test, n_train))

for i in tqdm(range(n_test), desc="Calculating test-train distances"):
    for j in range(n_train):
        distance_matrix[i, j] = dtw_distance(X_test[i], X_train[j])

console.print("[green]Test-to-train distance matrix calculated.[/green]")


console.print("\n[bold]Training and predicting with k-NN...[/bold]")

knn_clf = KNeighborsClassifier(n_neighbors=4, metric="precomputed")

train_distance_matrix = np.zeros((n_train, n_train))
for i in tqdm(range(n_train), desc="Calculating train-train distances for .fit()"):
    for j in range(i, n_train):
        dist = dtw_distance(X_train[i], X_train[j])
        train_distance_matrix[i, j] = dist
        train_distance_matrix[j, i] = dist

knn_clf.fit(train_distance_matrix, y_train)
console.print("[green]Model training complete.[/green]")

y_pred = knn_clf.predict(distance_matrix)
console.print("[green]Prediction complete.[/green]")


console.print("\n[bold]Estimation and saving results...[/bold]")

accuracy = accuracy_score(y_test, y_pred)

report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

conf_matrix = confusion_matrix(y_test, y_pred)

console.print(f"\n[bold]Accuracy: [cyan]{accuracy:.4f}[/cyan] ({accuracy * 100:.2f}%)")
console.print("\n[bold]Classification Report:[/bold]")
console.print(report)


timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
accuracy_str = f"acc_{accuracy:.4f}".replace(".", "_")
results_folder_name = f"{timestamp}_{accuracy_str}"
RESULTS_DIR = os.path.join(
    "data/training_results/10_dtw_knn_results", results_folder_name
)

os.makedirs(RESULTS_DIR, exist_ok=True)
console.print(f"\n[yellow]Results will be saved in: {RESULTS_DIR}[/yellow]")

with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n")
    f.write("Classification Report:\n")
    f.write(report)
console.print("[green]Classification report saved.[/green]")


plt.figure(figsize=(12, 10))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
)
plt.title(f"Confusion Matrix - DTW+kNN (k=4)\nAccuracy: {accuracy * 100:.2f}%")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
console.print("[green]Confusion matrix plot saved.[/green]")

console.print("\n[bold green]>>> FINISHED <<<[/bold green]\n")
plt.show()
