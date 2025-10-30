"""
goodreads_text_features_trial.py
--------------------------------
Trial SageMaker Processing Job Script
This script performs a small-scale trial run to validate the SageMaker Processing
environment and data paths. It reads a limited number of rows (default = 2000)
from the Goodreads dataset stored in Parquet format, and writes the sampled
subset back to S3 to confirm that input/output configurations and IAM permissions
are working correctly.
"""

import pandas as pd
import os

# -----------------------------------------------------------------------------
# Step 1: Define environment variables for input/output paths and sampling size
# -----------------------------------------------------------------------------
# The environment variables INPUT_DIR, OUTPUT_DIR, and ROW_CHUNK are set by the
# SageMaker Processing job. If they are not found, default local paths are used.
# This design keeps the script flexible and avoids hard-coded S3 paths.

input_dir = os.environ.get("INPUT_DIR", "/opt/ml/processing/input/features")
output_dir = os.environ.get("OUTPUT_DIR", "/opt/ml/processing/output")
row_chunk = os.environ.get("ROW_CHUNK", "2000")  # default: sample first 2000 rows

# -----------------------------------------------------------------------------
# Step 2: Load dataset from Parquet
# -----------------------------------------------------------------------------
# Pandas automatically reads a single Parquet file or merges multiple Parquet
# files within the same directory. This is typical for DataBrew or SageMaker
# outputs where data is sharded.

df = pd.read_parquet(input_dir)

# -----------------------------------------------------------------------------
# Step 3: Select a small subset for the trial
# -----------------------------------------------------------------------------
# Only the first N rows are taken. This small batch verifies that the pipeline
# functions correctly before running the full-scale feature engineering job.

df_subset = df.head(int(row_chunk))

# -----------------------------------------------------------------------------
# Step 4: Save the trial subset to output directory
# -----------------------------------------------------------------------------
# The subset is saved as 'trial_output.parquet' inside the output directory.
# SageMaker automatically uploads this file to the configured S3 destination
# when the job completes successfully.

output_path = f"{output_dir}/trial_output.parquet"
df_subset.to_parquet(output_path, index=False)

# -----------------------------------------------------------------------------
# Step 5: Log confirmation message
# -----------------------------------------------------------------------------
# Printing to stdout ensures that the SageMaker job logs show clear information
# on how many rows were loaded and saved. Useful for debugging and validation.

print(f"Loaded {len(df)} rows. Saved {len(df_subset)} rows to {output_path}.")
