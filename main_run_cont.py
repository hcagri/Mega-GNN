import subprocess
import time
import psutil  # To check running processes

# Global variable to store the PID of the last started process
last_pid = None

def is_pid_running(pid):
    """Check if a process with the given PID is still running."""
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True  # Assume it's running if we can't access it
    except Exception as e:
        print(f"Error checking PID {pid}: {e}")
        return False

def run_training():
    """Start the training script."""
    try:
        # Your command to run the training script
        command = [
            "python", "main.py", 
            "--data", "ETH-Kaggle", 
            "--model", "gin", 
            "--emlps", 
            "--ego", 
            "--reverse_mp",
            "--flatten_edges", 
            "--edge_agg_type", "genagg",
            "--task", "node_class",
            "--n_epochs", "80"
        ]

        # Run the command in the background using Popen
        process = subprocess.Popen(command)
        print(f"Started training with PID: {process.pid}")
        return process.pid  # Return the PID of the started process
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def main():
    global last_pid  # Access the global PID variable
    job_count = 0
    while job_count < 5:

        if last_pid is None or not is_pid_running(last_pid):
            print("\n\n\n\n\n -------------- New Training is Started -------------- \n\n\n\n\n")
            last_pid = run_training()  # Start the new process and store its PID
            job_count+=1

        # Check every 30 seconds (or adjust the time if needed)
        time.sleep(10*60) # wait for 10 minutes before checking again if the process still continues.

if __name__ == "__main__":
    main()
