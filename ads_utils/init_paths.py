# init_paths.py
import os
import sys
import platform

AddaxAI_files = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.realpath(__file__)))))

# insert dependencies to system variables
cuda_toolkit_path = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
paths_to_add = [
    # os.path.join(AddaxAI_files),
    # os.path.join(AddaxAI_files, "cameratraps"),
    # os.path.join(AddaxAI_files, "cameratraps", "megadetector"),
    # os.path.join(AddaxAI_files, "AddaxAI", "streamlit-AddaxAI"),
    # os.path.join(AddaxAI_files, "AddaxAI")
]
if cuda_toolkit_path:
    paths_to_add.append(os.path.join(cuda_toolkit_path, "bin"))
for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)

# Update PYTHONPATH env var without duplicates
PYTHONPATH_separator = ":" if platform.system() != "Windows" else ";"
existing_paths = os.environ.get("PYTHONPATH", "").split(PYTHONPATH_separator)

# Remove empty strings, duplicates, and keep order
existing_paths = [p for i, p in enumerate(
    existing_paths) if p and p not in existing_paths[:i]]

for path in paths_to_add:
    if path not in existing_paths:
        existing_paths.append(path)

os.environ["PYTHONPATH"] = PYTHONPATH_separator.join(existing_paths)
