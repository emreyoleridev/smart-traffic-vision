import os
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN")
repo_id = "emreyoleridev/smart-traffic-vision"

api = HfApi(token=token)

try:
    # Use "gradio" just to satisfy the server's strict create_repo check, 
    # the README.md will override this to streamlit.
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        private=False,
        exist_ok=True
    )
    print(f"Space {repo_id} created or already exists.")
except Exception as e:
    print(f"Error creating space: {e}")

try:
    api.upload_folder(
        folder_path=".",
        repo_id=repo_id,
        repo_type="space",
        token=token,
        ignore_patterns=[".git", ".venv", "__pycache__", "deploy_to_hf.py", ".gitignore"]
    )
    print("Files uploaded successfully! The README.md will correctly configure it as a Streamlit space.")
except Exception as e:
    print(f"Failed to upload files: {e}")

print(f"Deployment complete! View your space at: https://huggingface.co/spaces/{repo_id}")
