# Instructions to Run Mitra

Before building the image make sure the `weights` are present in the same folder where TabRepo is located - files should be in the `weights` folder.
This version of `flash-attn` only supports Ampere or newer GPUs (A100 or L4 arch and above), does not support T4 GPUs.
Hence, use only G6 or above to benchmark. huggingface/transformers#28188

If you want to use older GPUs consider downgrading the `flash-attn` version.

After building the image, add/change to your image uri in `tabrepo/tabflow/scripts/run_jobs_mitra_prateek_example_2025_06_05.py` and run the script.