import os

import cv2
import numpy as np
import pickle
import pandas as pd
from mediapipe.python.solutions.pose import Pose
from rich.console import Console
from tqdm import tqdm

PROCESSED_VIDEOS_30FPS_DIR = "data/processed_videos_30fps/"
ANNOTATIONS_CSV_PATH = "data/annotations_csv/combined_annotations.csv"
OUTPUT_DIR = "data/processed_features/"
OUTPUT_FILENAME = "00_extracted_features.pkl"

console = Console()

pose = Pose(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.5,
)

annotations_df = pd.read_csv(ANNOTATIONS_CSV_PATH)

total_rows = len(annotations_df)
console.print(
    f"\n[bold cyan]>>> Starting processing {total_rows} annotations... <<<[/bold cyan]\n"
)

sequences = []
labels = []

for index, row in tqdm(
    annotations_df.iterrows(), total=total_rows, desc="Annotation analysis"
):
    video_filename = row["fileName"]
    if not video_filename.endswith(".mp4"):
        video_filename += ".mp4"

    start_frame = int(row["startFrame"])
    end_frame = int(row["endFrame"])
    label = row["label"]

    video_path = os.path.join(PROCESSED_VIDEOS_30FPS_DIR, video_filename)

    if not os.path.exists(video_path):
        console.print(
            f"[bold red]Missing video file:[/bold red] [yellow]{video_path}[/yellow]"
        )
        continue

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        console.print(
            f"[bold red]Cannot open video file:[/bold red] [yellow]{video_path}[/yellow]"
        )
        continue

    cap.set(
        cv2.CAP_PROP_POS_FRAMES, start_frame
    )  # Set specific start frame with `CAP_PROP_POS_FRAMES`

    current_frame_num = start_frame

    sequence_landmarks = []

    while current_frame_num <= end_frame:
        success, frame = cap.read()  # Reading one frame from video
        if not success:
            break

        image_rgb = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )  # Changing colours format because CV2 using BGR, but MediaPipe require RGB

        results = pose.process(image_rgb)

        # Checking if a person was detected in this frame, MediaPipe detect only the closest person
        if results.pose_world_landmarks:
            landmarks = results.pose_world_landmarks.landmark

            # Taking pose_world_landmarks coordinates (33 points, each with x, y, z, visibility), flatten them into one long list of 132 numbers
            frame_landmarks = np.array(
                [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]
            ).flatten()
            sequence_landmarks.append(frame_landmarks)

        current_frame_num += 1

    cap.release()

    # Adding the entire collected sequence to the main sequences list, and also label to the labels list
    if sequence_landmarks:
        sequences.append(
            np.array(sequence_landmarks).tolist()
        )  # Converting back to list for easier saving with pickle
        labels.append(label)

console.print(
    f"\n[bold green]Processing ended.\nProcessed {len(sequences)} sequences.[/bold green]\n"
)

console.print("[yellow]Saving data to a file...[/yellow]")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

output_path_pkl = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

data_to_save = {"sequences": sequences, "labels": labels}

with open(output_path_pkl, "wb") as f:
    pickle.dump(data_to_save, f)

console.print(
    f"\n[bold green]Data was successfully saved to the file:[/bold green] [underline cyan]{output_path_pkl}[/underline cyan]\n"
)
