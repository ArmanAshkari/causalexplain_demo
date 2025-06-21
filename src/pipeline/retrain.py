from math import ceil
import pickle
from src.taskmonitor.task_monitor import TaskMonitor, parse_args as task_monitor_parse_args
from src.util.directories import SCRATCH_DIR
import argparse


def submit_job(cluster, max_jobs, min_cycle, indices_file_dir, main_script_args):
    """
    """
    if cluster == 'notchpeak':
        arg_list = ["--cluster", "notchpeak",
                "--num_processor", "16"]
    elif cluster == 'kingspeak':
        arg_list = ["--cluster", "kingspeak",
                "--num_processor", "8"]
    
    arg_list.extend(["--run_script", "./src/pipeline/run.sh",
                "--py_module", "src.retrain.retrain",
                "--indices_file", indices_file_dir,
                "--main_script_args", main_script_args])
        
    args = task_monitor_parse_args(arg_list)
    
    with open(args.indices_file, 'rb') as f:
        arr = pickle.load(f)
        indices = list(range(len(arr)))

    num_cycle = max(ceil(len(indices)/(2 * max_jobs * args.num_processor)), min_cycle)

    task_monitor = TaskMonitor(cluster=args.cluster, workload='retrain', num_processor=args.num_processor, num_cycle=num_cycle, run_script=args.run_script, py_module=args.py_module, indices=indices, main_script_args=args.main_script_args)
    task_monitor.monitor()


def retrain(args):
    """
    """
    dataset_name = args.dataset
    model = args.model
    min_frequency = args.min_frequency
    max_support = args.max_support

    threshold_str = f'freq_{min_frequency}_supp_{max_support}'
    rules_dir = f'{SCRATCH_DIR}/data/{dataset_name}/rules_{threshold_str}.pkl'

    main_script_args = f'--dataset {dataset_name} --model {model} --threshold_str {threshold_str}'
    submit_job(cluster=args.cluster, max_jobs=args.max_jobs, min_cycle=args.min_cycle, indices_file_dir=rules_dir, main_script_args=main_script_args)


def parse_args():
    parser = argparse.ArgumentParser(description="Script to process a given Process ID (PID).")
    
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--cluster", type=str, help="Cluster name")
    parser.add_argument("--max_jobs", type=int, default=500, help="Maximum number of jobs")
    parser.add_argument("--min_cycle", type=int, default=1, help="Minimum number of cycles")
    parser.add_argument("--min_frequency", type=int, help="Minimum frequency of a rule to be considered. It must be an integer.")
    parser.add_argument("--max_support", type=float, help="Maximum support of a rule to be considered. It must be a float.")
 
    return parser.parse_args()


if __name__=='__main__':
    """
    time python -m src.pipeline.retrain --dataset so --model nn --cluster kingspeak --min_frequency 1000 --max_support 0.3 --max_jobs 500

    time python -m src.pipeline.retrain --dataset compas --model linsvc --cluster kingspeak --min_frequency 1000 --max_support 0.3 --max_jobs 20
    """
    args = parse_args()
    retrain(args)