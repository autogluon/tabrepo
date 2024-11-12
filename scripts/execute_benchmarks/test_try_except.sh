#!/usr/bin/bash

set -e
log_dir="amlb_logs"

mkdir -p $log_dir

#log_error="$log_dir/error.txt"
#exec 2>"$log_error"
#log_out="$log_dir/output.txt"
#exec 1>"$log_out"
error_log_file="$log_dir/error.log"

# trap 'aws s3 cp "${log_dir}" {s3_output}/setup_logs --recursive' EXIT
trap 'ls "${log_dir}" -l' EXIT
trap 'ls $log_dir -l' EXIT

err_report() {
    err_code=$?
    err_line=$1
    err_command=$BASH_COMMAND

    echo -e "Bash Error Occurred...\nLine: ${err_line}\nError Code: ${err_code}\nCommand: ${err_command}"  # \nError Message: $(< "$log_error")" > "$error_log_file"
    # exit $err_code
}

final() {
    err_code=$?
    err_line=$1
    err_command=$BASH_COMMAND

    echo "Exiting! Exit Code: ${err_code}"
    # aws s3 cp "$log_dir" '{s3_output}/setup_logs' --recursive
    # exit $err_code
}

trap 'err_report $LINENO' ERR
# trap 'final' EXIT


ls
# ls asdaasdasd
echo "hello"

python run_except.py

echo "hi"

# ls asdasasdads
