import argparse
import os
import signal


def kill_process_group(pgid):
    """
    Kill all processes in the given process group.
    
    Args:
        pgid (int): The Process Group ID (PGID) to terminate.
    """
    try:
        # Send SIGTERM to all processes in the group
        os.killpg(pgid, signal.SIGTERM)
        print(f"Sent SIGTERM to process group {pgid}")
    except ProcessLookupError:
        print(f"Process group {pgid} does not exist or is already terminated.")
    except PermissionError:
        print(f"No permission to terminate process group {pgid}.")

    # If processes are still running, send SIGKILL
    try:
        os.killpg(pgid, signal.SIGKILL)
        print(f"Sent SIGKILL to process group {pgid}")
    except ProcessLookupError:
        print(f"Process group {pgid} is no longer running.")
    except PermissionError:
        print(f"No permission to forcibly terminate process group {pgid}.")


def parse_args():
    parser = argparse.ArgumentParser(description="Script to process a given Process ID (PID).")
    parser.add_argument("--pgid", type=int, help="The process group ID (PGID) to be processed. It must be an integer.")
    return parser.parse_args()


if __name__ == "__main__":
    """
    time python -m src.util.terminate_pgrp --pgid 12345
    """
    # Replace this with the PGID you want to terminate
    args = parse_args()
    # pgid_to_kill = int(input("Enter the Process Group ID (PGID) to kill: "))
    kill_process_group(args.pgid)