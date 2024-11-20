from huggingface_hub import snapshot_download

snapshot_download(repo_id="SoccerNet/SN-GSR-2025",
                  repo_type="dataset", revision="main",
                  local_dir="/Users/kai/GSR/data/SN-GSR-2025")
