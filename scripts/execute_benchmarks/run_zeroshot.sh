#!/bin/bash

# Install code remotely
set -e

echo "Working directory: $PWD"

# Verify the runbenchmark.py file exists in the expected location
FILE_RUNBENCHMARK="../automlbenchmark/runbenchmark.py"
if [ -f "$FILE_RUNBENCHMARK" ]; then
    echo "$FILE_RUNBENCHMARK exists."
else
    echo "ERROR: $FILE_RUNBENCHMARK does not exist."
    exit 1
fi

# Copy the custom configs to the current directory so we can specify them in the runbenchmark.py arguments
cp -r ../automlbenchmark/custom_configs/ ./

# Run the full TabRepo experiments
# Wait 6500 seconds between each command
#  6500 seconds due to 8 seconds between instance startup, and needing to start 244*3 instances = 5856 seconds, plus some overhead
#  117 commands to execute, for a total of 6500*117 seconds = 760500 seconds = 211 hours = 9 days
SLEEP_TIME=6500
FILES=(
  "python ../automlbenchmark/runbenchmark.py AutoGluon_bq:latest ag_244 1h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py AutoGluon_hq:latest ag_244 1h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py AutoGluon_mq:latest ag_244 1h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py H2OAutoML:2023Q2 ag_244 1h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py autosklearn:2023Q2 ag_244 1h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py autosklearn2:2023Q2 ag_244 1h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py flaml:2023Q2 ag_244 1h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py GAMA_benchmark:2023Q2 ag_244 1h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py lightautoml:2023Q2 ag_244 1h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py mljarsupervised_benchmark:2023Q2 ag_244 1h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py constantpredictor:2023Q2 ag_244 1h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py RandomForest:2023Q2 ag_244 1h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py TunedRandomForest:2023Q2 ag_244 1h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py AutoGluon_bq:latest ag_244 4h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py AutoGluon_hq:latest ag_244 4h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py AutoGluon_mq:latest ag_244 4h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py H2OAutoML:2023Q2 ag_244 4h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py autosklearn:2023Q2 ag_244 4h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py autosklearn2:2023Q2 ag_244 4h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py flaml:2023Q2 ag_244 4h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py GAMA_benchmark:2023Q2 ag_244 4h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py lightautoml:2023Q2 ag_244 4h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py mljarsupervised_benchmark:2023Q2 ag_244 4h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py constantpredictor:2023Q2 ag_244 4h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py RandomForest:2023Q2 ag_244 4h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py TunedRandomForest:2023Q2 ag_244 4h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_catboost_b0:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_catboost_b1:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_catboost_b2:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_catboost_b3:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_catboost_b4:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_catboost_b5:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_catboost_b6:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_catboost_b7:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_catboost_b8:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_catboost_b9:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_catboost_b10:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_catboost_b11:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_catboost_b12:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_lightgbm_b0:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_lightgbm_b1:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_lightgbm_b2:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_lightgbm_b3:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_lightgbm_b4:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_lightgbm_b5:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_lightgbm_b6:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_lightgbm_b7:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_lightgbm_b8:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_lightgbm_b9:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_lightgbm_b10:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_lightgbm_b11:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_lightgbm_b12:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_xgboost_b0:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_xgboost_b1:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_xgboost_b2:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_xgboost_b3:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_xgboost_b4:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_xgboost_b5:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_xgboost_b6:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_xgboost_b7:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_xgboost_b8:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_xgboost_b9:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_xgboost_b10:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_xgboost_b11:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_xgboost_b12:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_fastai_b0:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_fastai_b1:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_fastai_b2:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_fastai_b3:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_fastai_b4:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_fastai_b5:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_fastai_b6:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_fastai_b7:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_fastai_b8:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_fastai_b9:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_fastai_b10:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_fastai_b11:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_fastai_b12:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_nn_torch_b0:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_nn_torch_b1:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_nn_torch_b2:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_nn_torch_b3:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_nn_torch_b4:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_nn_torch_b5:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_nn_torch_b6:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_nn_torch_b7:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_nn_torch_b8:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_nn_torch_b9:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_nn_torch_b10:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_nn_torch_b11:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_nn_torch_b12:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_rf_b0:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_rf_b1:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_rf_b2:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_rf_b3:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_rf_b4:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_rf_b5:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_rf_b6:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_rf_b7:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_rf_b8:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_rf_b9:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_rf_b10:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_rf_b11:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_rf_b12:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_xt_b0:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_xt_b1:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_xt_b2:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_xt_b3:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_xt_b4:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_xt_b5:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_xt_b6:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_xt_b7:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_xt_b8:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_xt_b9:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_xt_b10:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_xt_b11:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
  "python ../automlbenchmark/runbenchmark.py ZS_BAG_xt_b12:zeroshot ag_244 60h8c -u custom_configs -f 0 1 2 -m aws -p 3000"
)

i=0
for f in "${FILES[@]}"; do
  echo "==================================="
  echo "Executing command: $f"
  ((i=i+1))
  LOG_FILE_NAME="log_$i.file"
  LOG_FILE_NAME=$(echo "$LOG_FILE_NAME" | tr '/\' _)  # Remove / and \
  nohup $f > $LOG_FILE_NAME 2>&1 &
  echo "Command executed, logging to ${LOG_FILE_NAME}, sleeping for ${SLEEP_TIME}s..."

  sleep $SLEEP_TIME
done


echo "==================================="

echo "Done!"

exit 0
