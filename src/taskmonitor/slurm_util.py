import argparse
import subprocess
import re


def parse_args(arg_string):
    """
    """
    parser = argparse.ArgumentParser(description="Automated Job submission.")

    # Add optional arguments with default values
    parser.add_argument("--job_name", type=str, default="test", help="Job name")
    parser.add_argument("--script", type=str, default="run.sh", help="Script you want to run")
    parser.add_argument("--array_index", type=str, default="0-0", help="Array index")
    parser.add_argument("--cluster", type=str, default="notchpeak", help="Cluster name")
    parser.add_argument("--account", type=str, default="owner-guest", help="Account")
    parser.add_argument("--partition", type=str, default="notchpeak-shared-guest", help="Partition")
    parser.add_argument("--time_limit", type=str, default="24:00:00", help="Time limit (hr::mm:ss)")
    parser.add_argument("--num_nodes", type=str, default="1", help="Number of nodes")
    parser.add_argument("--num_tasks", type=str, default="1", help="Number of tasks")
    parser.add_argument("--cpus_per_task", type=str, default="16", help="CPUs per task")
    parser.add_argument("--memory", type=str, default="32", help="Memory allocation (GB)")
    
    # Add GPU flag; if specified, store "gpu:1", otherwise store an empty string
    parser.add_argument("--gpu", action='store_const', const="gpu:1", default="", help="Enable GPU (if any)")

    return parser.parse_args(arg_string.split())


def submit_job(job_submit_args, job_args, main_script_args):
    """
    """
    args = parse_args(job_submit_args)
    job_submit_script_path = './src/taskmonitor/job_array_template.sh'
    
    # ****** ORDER IS IMPORTANT. MUST MATCH WITH JOB SUBMISSION SCRIPT ******
    arguments = [args.job_name, 
                 args.script, 
                 args.array_index,
                 args.cluster, 
                 args.account, 
                 args.partition,
                 args.time_limit, 
                 args.num_nodes, 
                 args.num_tasks, 
                 args.cpus_per_task, 
                 args.memory, 
                 args.gpu]

    command = [job_submit_script_path] + arguments + job_args.split() + main_script_args.split()
    # print(command)

    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("Output:", result.stdout)
        
        if result.stderr:
            print("Error:", result.stderr)

    except subprocess.CalledProcessError as e:
        print(f"Script failed with error: {e}")
    
    last_output = result.stdout.splitlines()[-1]
    pattern = r"Job (\d+)"
    match = re.search(pattern, last_output)
    return match.group(1) if match else None


def cancel_job(jobid):
    """
    """
    try:
        result = subprocess.run(['scancel', jobid], capture_output=True, text=True, check=True)
        print(f"Job {jobid} successfully canceled.")
    
    except subprocess.CalledProcessError as cpe:
        # This exception is raised if check=True and the command returns a non-zero exit code
        print(f"Failed to cancel job {jobid}. Command returned non-zero exit code: {cpe.returncode}")
        print(f"Error: {cpe.stderr}")