import pickle
import numpy as np
from scipy.io import savemat
from rich.console import Console

console = Console()

NORMALIZED_DATA_PATH = "data/processed_features/normalization_with_geometrical_mean_calc/01_normalized_sequences_with_legs_pose_world_landmarks_image_mode_false_model_complexity_1.pkl"
OUTPUT_MAT_PATH = "data/matlab_data_for_LDMLT.mat"

console.print(f"Loading data from [cyan]{NORMALIZED_DATA_PATH}[/cyan]...")

try:
    with open(NORMALIZED_DATA_PATH, "rb") as f:
        data = pickle.load(f)

    sequences = data["sequences"]
    labels = data["labels"]

    sequences_for_matlab = np.empty((len(sequences), 1), dtype=object)
    for i, seq in enumerate(sequences):
        sequences_for_matlab[i, 0] = np.array(seq, dtype=np.float64)

    labels_for_matlab = np.empty((len(labels), 1), dtype=object)
    for i, label in enumerate(labels):
        labels_for_matlab[i, 0] = label

    matlab_data = {"sequences": sequences_for_matlab, "labels": labels_for_matlab}

    console.print(f"Saving data to [cyan]{OUTPUT_MAT_PATH}[/cyan]...")
    savemat(OUTPUT_MAT_PATH, matlab_data)

    console.print("[bold green]Conversion successful[/bold green]")

except FileNotFoundError:
    console.print(
        "[bold red]ERROR: File not found. Make sure the .pkl file exists.[/bold red]"
    )
