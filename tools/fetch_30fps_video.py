import os
import subprocess

import yt_dlp
from rich.console import Console

console = Console()

video_urls = [
    "https://www.youtube.com/watch?v=Tcly8G0MQDg",  # Novak Djokovic & Jack Sock | IW Practice (Court Level 60fps)
    "https://www.youtube.com/watch?v=DGIkkdxMrWc",  # Novak Djokovic Court Level Practice 2023 | Groundstrokes, Serves, Volleys + Warmup (4K 60FPS)
    "https://www.youtube.com/watch?v=ZVj-ukiIDJE",  # Carlos Alcaraz INTENSE practice with Pablo Carreño Busta | Miami Open 2022
    "https://www.youtube.com/watch?v=kTO7o5A0sys",  # Carlos Alcaraz | Court Level Practice [2023 IW]
    "https://www.youtube.com/watch?v=gRSRlMZbVYE",  # Taylor Fritz Court Level Practice | Australian Open (4K 60FPS)
]

base_dir = os.path.abspath("data")
originals_dir = os.path.join(base_dir, "raw_videos")
converted_dir = os.path.join(base_dir, "processed_videos_30fps")

os.makedirs(originals_dir, exist_ok=True)
os.makedirs(converted_dir, exist_ok=True)


#! FFmpeg should be installed in system and available in SYSTEM PATH
def convert_to_30fps(input_path, output_folder, video_id):
    """Converts a video file to 30 FPS using FFmpeg."""
    output_filename = f"{video_id}_30fps.mp4"
    output_path = os.path.join(output_folder, output_filename)

    cmd = [
        "ffmpeg",
        "-i",
        input_path,
        "-r",
        "30",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "22",
        "-an",
        output_path,
    ]

    console.print(
        f"[yellow]Conversion:[/yellow] {os.path.basename(input_path)} [cyan]{output_filename}[/cyan]"
    )

    subprocess.run(cmd, check=True)

    return output_path


with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
    console.print(
        f"\n[bold]>>> Starting download and conversion {len(video_urls)} videos... <<<[/bold]\n"
    )

    for url in video_urls:
        console.print(f"[bold]URL processing:[/bold] {url}")
        try:
            video_id = url.split("v=")[-1]
            output_filename = f"{video_id}.mp4"
            output_path = os.path.join(originals_dir, output_filename)

            ydl_options = {
                "format": "bestvideo[height=1080]/bestvideo",
                "outtmpl": output_path,
                "merge_output_format": "mp4",
                "quiet": False,
            }

            with yt_dlp.YoutubeDL(ydl_options) as single_ydl:
                single_ydl.download([url])
                console.print("[green]Downloaded successfully[/green]")

            if not os.path.exists(output_path):
                console.print(
                    f"[bold red]Not found:[/bold red] [yellow]{output_path}[/yellow]"
                )
                raise FileNotFoundError(f"Not found: {output_path}")

            converted_file = convert_to_30fps(output_path, converted_dir, video_id)
            console.print(
                f"[bold green]Saved converted file in:[/bold green] [underline cyan]{converted_file}[/underline cyan]"
            )

        except Exception as e:
            console.print(
                f"[bold red]An error occurred while processing {url}:[/bold red] {e}"
            )

console.print("\n[bold green]>>> FINISHED <<<[/bold green]\n")
