from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="google/paligemma2-3b-mix-224",
    local_dir="./hf_cache/google/paligemma2-3b-mix-224",
    token="hf_xx",   # only needed if repo is private
)
