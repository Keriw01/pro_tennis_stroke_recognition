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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

console = Console()
console.print("\n[bold cyan]>>> STARTING DTW + k-NN WITH LOSO CV <<<\n")


console.print("[bold]Loading normalized data...[/bold]")

K = 18
DISTANCE = cosine
DISTANCE_NAME = "cosine"
PREFIX = f"k={K}_{DISTANCE_NAME}"
NORMALIZED_DATA_PATH = "data/processed_features/01_normalized_sequences_with_legs_pose_world_landmarks_image_mode_false_model_complexity_1.pkl"

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
subjects = []
labels = []

for seq_list, full_labels in zip(sequences_raw, labels_raw):
    if len(seq_list) >= MIN_SEQUENCE_LENGTH:
        sequences.append(np.array(seq_list, dtype=np.float64))
        parts = full_labels.split(" ")
        labels.append(parts[0])
        subjects.append(parts[1])

console.print(f"Filtered ended to [bold]{len(sequences)}[/bold] valid sequences.")


sequences = np.array(sequences, dtype=object)
labels = np.array(labels)
subjects = np.array(subjects)


console.print("\n[bold]Preparing labels and splitting data...[/bold]")

# Changing text labels to integer representation
label_encoder = LabelEncoder()
y_numeric = label_encoder.fit_transform(labels)

console.print("[underline]Mapped labels to numbers:[/underline]")
for i, class_name in enumerate(label_encoder.classes_):
    console.print(f"  [magenta]{i}[/magenta]: {class_name}")

unique_subjects = np.unique(subjects)
console.print(
    f"\nFound [bold]{len(unique_subjects)}[/bold] unique subjects for LOSO CV: {unique_subjects}"
)


console.print("\n[bold]Starting Leave-One-Subject-Out Cross-Validation...[/bold]")

all_y_test = []
all_y_pred = []
fold_accuracies = {}

for subject_to_leave_out in unique_subjects:
    console.print(
        f"\nTesting on Subject: [bold yellow]{subject_to_leave_out}[/bold yellow]"
    )

    test_mask = subjects == subject_to_leave_out
    train_mask = ~test_mask

    X_train, X_test = sequences[train_mask], sequences[test_mask]
    y_train, y_test = y_numeric[train_mask], y_numeric[test_mask]

    console.print(
        f"Training set size: [bold]{len(X_train)}[/bold], Test set size: [bold]{len(X_test)}[/bold]"
    )

    def dtw_distance(s1, s2):
        """Calculates the DTW distance between two sequences."""
        distance, _ = fastdtw(s1, s2, dist=DISTANCE)
        return distance

    n_test, n_train = len(X_test), len(X_train)

    distance_matrix = np.zeros((n_test, n_train))

    for i in tqdm(range(n_test), desc="Calculating test-train distances"):
        for j in range(n_train):
            distance_matrix[i, j] = dtw_distance(X_test[i], X_train[j])

    console.print("[green]Test-to-train distance matrix calculated.[/green]")

    train_distance_matrix = np.zeros((n_train, n_train))
    for i in tqdm(range(n_train), desc="Calculating train-train distances for .fit()"):
        for j in range(i, n_train):
            dist = dtw_distance(X_train[i], X_train[j])
            train_distance_matrix[i, j] = dist
            train_distance_matrix[j, i] = dist

    console.print("[green]Train-to-train distance matrix calculated.[/green]")

    console.print("\n[bold]Training and predicting with k-NN...[/bold]")

    knn_clf = KNeighborsClassifier(n_neighbors=K, metric="precomputed")

    knn_clf.fit(train_distance_matrix, y_train)
    console.print("[green]Model training complete.[/green]")

    y_pred = knn_clf.predict(distance_matrix)
    console.print("[green]Prediction complete.[/green]")

    fold_accuracy = accuracy_score(y_test, y_pred)
    fold_accuracies[subject_to_leave_out] = fold_accuracy
    all_y_test.extend(y_test)
    all_y_pred.extend(y_pred)

    console.print(
        f"[green]Accuracy for subject {subject_to_leave_out}: {fold_accuracy:.4f}[/green]"
    )

console.print("\n[bold]Aggregating and displaying final results...[/bold]")

all_y_test = np.array(all_y_test)
all_y_pred = np.array(all_y_pred)
accuracies_list = list(fold_accuracies.values())

mean_accuracy = np.mean(accuracies_list)
std_accuracy = np.std(accuracies_list)
console.print(
    f"\n[bold]Average LOSO CV Accuracy: [cyan]{mean_accuracy:.4f} ± {std_accuracy:.4f}[/cyan]"
)

console.print("\n[underline]Accuracy for each subject left out:[/underline]")
for subject, acc in fold_accuracies.items():
    console.print(f"  - {subject}: {acc:.4f}")

report = classification_report(
    all_y_test, all_y_pred, target_names=label_encoder.classes_
)
conf_matrix = confusion_matrix(all_y_test, all_y_pred)

console.print("\n[bold]Overall classification report (from all folds):[/bold]")
console.print(report)


timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
accuracy_str = f"acc_{mean_accuracy:.4f}".replace(".", "_")
results_folder_name = f"{PREFIX}_{timestamp}_{accuracy_str}"
RESULTS_DIR = os.path.join(
    "data/training_results/10_dtw_knn_results_hiperparameters", results_folder_name
)
os.makedirs(RESULTS_DIR, exist_ok=True)
console.print(f"\n[yellow]Results will be saved in: {RESULTS_DIR}[/yellow]")

with open(os.path.join(RESULTS_DIR, "detailed_accuracies.txt"), "w") as f:
    f.write(f"Average LOSO CV Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}\n\n")
    f.write("Accuracy for each subject left out:\n")
    for subject, acc in fold_accuracies.items():
        f.write(f"  - {subject}: {acc:.4f}\n")

with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
    f.write(
        f"Average LOSO CV Accuracy: {mean_accuracy:.4f} ({mean_accuracy * 100:.2f}%)\n\n"
    )
    f.write("Overall Classification Report (from all folds):\n")
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
plt.title(
    f"Aggregated Confusion Matrix (LOSO) - DTW+kNN\nAverage Accuracy: {mean_accuracy * 100:.2f}%"
)
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_loso.png"))
console.print("[green]LOSO confusion matrix plot saved.[/green]")

console.print("\n[bold green]>>> FINISHED <<<[/bold green]\n")
plt.show()
