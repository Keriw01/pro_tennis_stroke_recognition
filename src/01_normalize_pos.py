import numpy as np
import pickle
from rich.console import Console
from tqdm import tqdm

EXTRACTED_FEATURES_PATH = "data/processed_features/00_extracted_features.pkl"
OUTPUT_NORMALIZED_SEQUENCES_PATH = (
    "data/processed_features/01_normalized_sequences_pos.pkl"
)
JOINT_INDICES_TO_USE = [11, 12, 13, 14, 15, 16, 17, 18, 23, 24, 25, 26, 27, 28]
# without legs
# JOINT_INDICES_TO_USE = [11, 12, 13, 14, 15, 16, 17, 18]

console = Console()
console.print(
    "\n[bold cyan]>>> Starting normalization (Option 1: All Frame Center)... <<<[/bold cyan]\n"
)


try:
    with open(EXTRACTED_FEATURES_PATH, "rb") as f:
        data = pickle.load(f)

    sequences = data["sequences"]
    labels = data["labels"]
except FileNotFoundError:
    console.print(
        f"[bold red] ERROR: File '{EXTRACTED_FEATURES_PATH}' not found.[/bold red]"
    )


normalized_sequences = []
# Iteration after each individual stroke sequence (e.g. one forehand)
for seq in tqdm(sequences, desc="Normalization", unit="sequence"):
    normalized_seq = []

    # Single frame processing
    for frame_landmarks_list in seq:
        # `frame_landmarks_list` is a flattened list of 132 coordinates
        frame_landmarks = np.array(frame_landmarks_list)
        landmarks_reshaped = frame_landmarks.reshape((33, 4))

        selected_landmarks = landmarks_reshaped[JOINT_INDICES_TO_USE]

        # Calculate the geometric center of gravity of the selected landmarks (x, y, z)
        center_point = np.mean(selected_landmarks[:, :3], axis=0)

        normalized_landmarks = selected_landmarks.copy()
        # *Information about where the player was standing on the court is removed
        # *Only information about their body position remains
        normalized_landmarks[:, :3] = normalized_landmarks[:, :3] - center_point

        normalized_seq.append(normalized_landmarks.flatten())

    normalized_sequences.append(
        np.array(normalized_seq).tolist()
    )  # Converting back to list for easier saving with pickle

console.print("\n[bold green]>>> FINISHED <<<[/bold green]\n")


console.print("[yellow]Saving data to a file...[/yellow]")
data_to_save = {"sequences": normalized_sequences, "labels": labels}

with open(OUTPUT_NORMALIZED_SEQUENCES_PATH, "wb") as f:
    pickle.dump(data_to_save, f)

console.print(
    f"\n[bold green]Data was successfully saved to the file:[/bold green] [underline cyan]{OUTPUT_NORMALIZED_SEQUENCES_PATH}[/underline cyan]\n"
)
