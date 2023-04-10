from pathlib import Path
from time import time

import whisper


def get_file_paths(
    input_dir: str | Path, pattern: str = "*.m4a", recursive: bool = True
) -> list[Path]:
    input_path = Path(input_dir)

    if recursive:
        pattern = f"**/{pattern}"

    file_paths: list[Path] = [
        f
        for f in input_path.glob(pattern)
        if f.is_file() and not f.stem.startswith(".")
    ]

    return sorted(file_paths)


recordings_dir: Path = Path("recordings/")

recordings: list[Path] = get_file_paths(recordings_dir)

print(f"Found {len(recordings)} audio files to transcribe.\n")


for recording in recordings:
    # models = ["tiny.en", "base.en", "small.en", "medium.en"]
    # models = ["tiny.en", "base.en", "small.en"]
    # models = ["tiny", "base", "small"]
    # models = ["medium"]
    # models = ["tiny", "base", "small", "medium", "large"]
    models = ["medium", "large"]
    # models = ["small"]

    for c_model in models:
        print(f"Running whisper model ({c_model}) on {recording.name}...")
        start_time = time()

        model = whisper.load_model(c_model)
        result = model.transcribe(str(recording), fp16=False, language="English")

        run_time = round(time() - start_time, 2)
        print(
            f"Finished running whisper model ({c_model}) on {recording.name} ({run_time}s)"
        )

        save_path = (
            Path(str(recording).replace("recordings", "transcripts"))
            .with_name(f"{recording.stem}-{c_model}")
            .with_suffix(".txt")
        )

        save_path.mkdir(parents=True, exist_ok=True)

        save_path.write_text(result["text"])

        print(f"Transcript written to {save_path}")
        print()
