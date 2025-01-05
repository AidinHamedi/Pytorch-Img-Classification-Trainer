# Libs >>>
import os
import subprocess


# Main func >>>
def run_ruff():
    # Ensure Logs\others directory exists
    if not os.path.exists("Logs/others"):
        os.makedirs("Logs/others")

    # Define the log file paths
    lint_log_path = os.path.join("Logs/others", "ruff_lint_log.txt")
    format_log_path = os.path.join("Logs/others", "ruff_format_log.txt")

    # Define the ruff commands for linting and formatting
    lint_command = ["ruff", "check", ".", "--output-format=text"]

    format_command = ["ruff", "format", "."]

    # Run the lint command and capture output
    with open(lint_log_path, "w") as lint_log:
        lint_result = subprocess.run(lint_command, capture_output=True, text=True)
        lint_log.write(lint_result.stdout)
        if lint_result.stderr:
            lint_log.write(lint_result.stderr)

    # Run the format command and capture output
    with open(format_log_path, "w") as format_log:
        format_result = subprocess.run(format_command, capture_output=True, text=True)
        format_log.write(format_result.stdout)
        if format_result.stderr:
            format_log.write(format_result.stderr)

    # Print success or error messages
    if lint_result.returncode == 0:
        print("Linting completed successfully.")
    else:
        print(
            f"Linting encountered issues. Please check the log file for more details: {lint_log_path}"
        )

    if format_result.returncode == 0:
        print("Formatting completed successfully.")
    else:
        print(
            f"Formatting encountered issues. Please check the log file for more details: {format_log_path}"
        )


if __name__ == "__main__":
    run_ruff()
