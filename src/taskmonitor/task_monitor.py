from datetime import datetime
import math
import pickle
import sys
import time
from math import ceil
from collections import defaultdict
import numpy as np
from multiprocessing import Process, Queue
import queue
from .slurm_util import submit_job
from .job_monitor import JobMonitor
from .util import ProcessTable, TableEntry, search_tracker_folder, TRACKER_DIR
from .task_id import SQLiteDatabaseIndexCounter
from .dual_writer import DualWriter
import argparse

np.random.seed(42)


class TaskMonitor:
    REQUEUE_TIME = 180 # Must be greater than SB_THRSHLD, otherwise failed processes keep respawning infinitely
    REQUEUE_BUFFER_TIME = 60

    def __init__(self, cluster, workload, num_processor, num_cycle, run_script, py_module, indices, main_script_args):
        counter = SQLiteDatabaseIndexCounter()
        self.taskid = counter.get_next_index()
        self.cluster = cluster
        self.workload = workload
        self.num_processor = num_processor
        self.num_cycle = num_cycle
        self.run_script = run_script
        self.py_module = py_module
        self.indices = indices
        self.main_script_args = main_script_args
        self.job_table = dict()
        self.task_table = defaultdict(TableEntry)
        self.p_map = ProcessTable()
        self.recent_returns = set()
        self.requeue_candidates = set()
        self.last_submission_time = math.inf
        self.last_buffer_time = math.inf
        # sys.stdout = DualWriter(f'{TRACKER_DIR}/console_log/{self.taskid}.txt')


    def resubmission_remaining_time(self):
        """
        """
        return TaskMonitor.REQUEUE_TIME - (time.time() - self.last_submission_time)

    
    def buffer_remaining_time(self):
        """
        """
        return TaskMonitor.REQUEUE_BUFFER_TIME - (time.time() - self.last_buffer_time)

    
    @staticmethod
    def worker(taskid, cluster, workload, num_processor, num_cycle, run_script, py_module, indices, main_script_args, msgQ):
        """
        """
        if not indices:
            msgQ.put(('STOP',))
            return
    
        total = len(indices)
        
        if workload == 'retrain':
            batch_size = num_processor * num_cycle * 2 # multiplied by 2 for hyperthreading
        elif workload == 'job_array':
            batch_size = 1
        
        job_arr_size = ceil(total / batch_size)
        max_simul_job = 1000 # job_arr_size # default_value

        partition = 'notchpeak-shared-guest'
        if cluster == 'kingspeak':
            partition = 'kingspeak-shared-guest'
        elif cluster == 'lonepeak':
            partition = 'lonepeak-shared-guest'

        jobid = submit_job(job_submit_args=f'--array_index 0-{job_arr_size-1}%{max_simul_job} \
                                            --cpus_per_task {num_processor} \
                                            --memory {num_processor * 4} \
                                            --script {run_script} \
                                            --cluster {cluster} \
                                            --partition {partition}',
                            job_args=f'{taskid} {py_module} {batch_size}',
                            main_script_args=main_script_args)
        
        with open(f'{TRACKER_DIR}/indices/{taskid}-{jobid}.pkl', 'wb') as f:
            pickle.dump(indices, f)
        
        # sys.stdout = DualWriter(f'{TRACKER_DIR}/console_log/{taskid}-{jobid}.txt', file_only=True) # Created inside a separate process and address spcae

        jobid_indices_map = {f'{jobid}_{i}': indices[i*batch_size:(i+1)*batch_size] for i in range(job_arr_size)}

        # print(jobid_indices_map)
        
        msgQ.put(('BEGIN', jobid, jobid_indices_map))

        job_monitor = JobMonitor(taskid=taskid, jobid=jobid, cluster=cluster, job_arr_size=job_arr_size, jobid_indices_map=jobid_indices_map, total=total, msgQ=msgQ)

        time.sleep(5)

        job_monitor.track_job_arr()
        
        msgQ.put(('STOP',))
        return


    def toggle_counter(self, buffer_on):
        """
        """
        if buffer_on:
            self.last_submission_time = math.inf
            self.last_buffer_time = time.time()
        else:
            self.last_submission_time = time.time()
            self.last_buffer_time = math.inf

    
    def create_process(self, indices):
        """
        """
        self.toggle_counter(buffer_on=False)
        msgQ = Queue()
        p = Process(target=TaskMonitor.worker, args=(self.taskid, self.cluster, self.workload, self.num_processor, self.num_cycle, self.run_script, self.py_module, indices, self.main_script_args, msgQ))
        self.p_map.add_process(p, msgQ)
        return p


    def update_job_mapping(self, jobid_indices_map):
        """
        """
        for jobid, indices in jobid_indices_map.items():
            for i in indices:
                assert i in self.task_table, 'Invalid index'
                self.task_table[i].get('JobAssignment').append(jobid)
        
        assert len(set(self.job_table.keys()) & set(jobid_indices_map.keys())) == 0, 'Reentry jobid map'
        self.job_table.update(jobid_indices_map)


    def update_recent_returns(self, jobid, RT_jobs):
        """
        """
        returned = set()
        for id_ in RT_jobs:
            assert id_.split('_')[0] == jobid, 'Jobid mismatch'
            returned.update(self.job_table[id_])
        
        self.recent_returns.update(returned)
    

    def move_returns_to_buffer(self):
        """
        """
        self.requeue_candidates = self.recent_returns
        self.recent_returns = set()
        self.toggle_counter(buffer_on=True)

    
    def get_requeue_list(self):
        """
        """
        complete = set()
        for id_ in self.p_map.get_all_ids():
            p, msgQ, jobid = self.p_map.get_process_data(id_)
            temp = set(search_tracker_folder(self.taskid, jobid))
            assert len(complete & temp) == 0, 'Double completion'
            complete.update(temp)

        for i in (self.requeue_candidates & complete):
            assert self.task_table[i].get('State') == 'INC', 'Double completion'
            self.task_table[i].set('State', 'CMPLT')
        
        requeue_list = list(self.requeue_candidates - complete)
        self.requeue_candidates = set()
        return requeue_list


    def monitor(self):
        """
        """
        for i in self.indices:
            self.task_table[i].set('State', 'INC')
            self.task_table[i].set('JobAssignment', [])
        
        p = self.create_process(self.indices)
        p.start()

        while True:
            p_ids = self.p_map.get_all_live_ids()
            
            if not p_ids and not self.recent_returns and not self.requeue_candidates:
                break
            
            print('\n',
                  datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '\n',
                  'taskid', self.taskid, '\n',
                  'p_ids', p_ids, '\n',
                  'p_map', self.p_map, '\n',
                  'recent_returns', f'{len(self.recent_returns)} entries', '\n',
                  'requeue_candidates', f'{len(self.requeue_candidates)} entries', '\n',
                  f"{(len([k for k in self.task_table if self.task_table[k].get('State') == 'CMPLT']) * 100 / len(self.task_table)):.2f}% complete", '\n',
                  '\n')

            for id_ in p_ids:
                p, msgQ, jobid = self.p_map.get_process_data(id_)
                try:
                    msg = msgQ.get_nowait()
                    print(f'\n***{id_}:{jobid} msgQ has msg: {msg[0]}***\n')
                    if msg[0] == 'BEGIN':
                        jobid, jobid_indices_map = msg[1], msg[2]
                        self.p_map.add_jobid(id_, jobid)
                        self.update_job_mapping(jobid_indices_map)

                    elif msg[0] == 'STOP':
                        self.p_map.kill_process(id_)
                        print(f'\n***Killing process {id_}:{jobid}***\n')

                    elif msg[0] == 'REPORT':
                        RT_jobs = msg[1]
                        self.update_recent_returns(jobid, RT_jobs)                     
                except queue.Empty:
                    pass
            
            if self.resubmission_remaining_time() <= 0:
                self.move_returns_to_buffer()
            
            if self.buffer_remaining_time() <= 0:
                requeue_list = self.get_requeue_list()            
                p = self.create_process(requeue_list)
                p.start()
            
            time.sleep(10)
                
        for id_ in self.p_map.get_all_ids():
            p, msgQ, jobid = self.p_map.get_process_data(id_)
            p.join()

        # for k in self.task_table.keys():
        #     print(f"{k}: {self.task_table[k].get('JobAssignment')}")

        time.sleep(TaskMonitor.REQUEUE_BUFFER_TIME)
        
        self.final_check()

        print('Success')

    
    def final_check(self):
        """
        """
        completed = set()
        for id_ in self.p_map.get_all_ids():
            p, msgQ, jobid = self.p_map.get_process_data(id_)
            
            if jobid == -1: # dummy process
                continue

            temp = set(search_tracker_folder(taskid=self.taskid, jobid=jobid))
            
            if (completed & temp): # Multiple instance
                assert False, 'Multiple instance'
            completed.update(temp)

            for i in self.task_table.keys():
                if i in temp: # Is in that folder
                    if not self.task_table[i].get('JobAssignment')[-1].startswith(jobid): # But it should NOT be there
                        assert False, "Should'nt be there, but is"
                else: # Is NOT in that folder
                    if self.task_table[i].get('JobAssignment')[-1].startswith(jobid): # But it should be there
                        assert False, "Should be there, but isn't"
        
        return True # Everything is where it should be


def parse_args(arg_list=None):
    parser = argparse.ArgumentParser(description="Parse arguments for running a script with specified module and indices.")
    
    # Add each argument as a required string argument
    parser.add_argument('--cluster', type=str, default="notchpeak", help="Cluster name")
    parser.add_argument("--num_processor", type=int, default=16, help="Number of processors")
    parser.add_argument('--run_script', type=str, required=True, help="The path or command to the script to be executed.")
    parser.add_argument('--py_module', type=str, required=True, help="The Python module to be run.")
    parser.add_argument('--indices_file', type=str, required=True, help="The path to Indices file.")
    parser.add_argument('--main_script_args', type=str, required=True, help="Additional arguments for the main script, provided as a string.")
    
    # Parse and return the arguments
    args = parser.parse_args() if arg_list is None else parser.parse_args(arg_list)
    return args


if __name__=="__main__":
    args = parse_args()
    
    with open(args.indices_file, 'rb') as f:
        test_set_indices = pickle.load(f)
        indices = list(range(len(test_set_indices)))

        # partial indices
        # indices = list(range(1000))
        
        # manual indices
        # indices = [4118, 4121, 4122, 4123, 4124, 4125, 4126, 4127, 4139, 4140, 4141, 4142, 4143, 4144, 4145, 4146, 4147, 2109, 2111, 2122, 2124, 2126, 2127, 4185, 4187, 4188, 2143, 4207, 4208, 4210, 4211, 4212, 4213, 4214, 4215, 4216, 4217, 4218, 4237, 2219, 2220, 2226, 2227, 2228, 2229, 2230, 2231, 2232, 185, 191, 2264, 2266, 2267, 2268, 2269, 2270, 2271, 2288, 2289, 2290, 2291, 2292, 2293, 2294, 2295, 2296, 266, 4363, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 2327, 4375, 2330, 2331, 2332, 2333, 2334, 4378, 4380, 4381, 4382, 4383, 4385, 4392, 4393, 4394, 4395, 4396, 4397, 4398, 4399, 4400, 4401, 4402, 4403, 2356, 2357, 2358, 2359, 2360, 4412, 4413, 4414, 4415, 4424, 4433, 4434, 4435, 2388, 4436, 2390, 4437, 2392, 2393, 2394, 2395, 2396, 4438, 4439, 4440, 4441, 4442, 4443, 2416, 2417, 2418, 2419, 2420, 2421, 2422, 2423, 4466, 4467, 4468, 4469, 4470, 4471, 4472, 4473, 4474, 4475, 4476, 4477, 4478, 4486, 2447, 2448, 2449, 2451, 2452, 2453, 2454, 2455, 2456, 2457, 2458, 2459, 2460, 2461, 2462, 4502, 4503, 4507, 4521, 4522, 4523, 4524, 4525, 4526, 4527, 4528, 4529, 2487, 2489, 2490, 2491, 2492, 2493, 4543, 4552, 4553, 4554, 4555, 4556, 4557, 4591, 4594, 4595, 4596, 4597, 4598, 4599, 4600, 4601, 4602, 4603, 4604, 4605, 2558, 2559, 4606, 4607, 4623, 4624, 4625, 4626, 4627, 4628, 4629, 4630, 4631, 4632, 4633, 4634, 4635, 4636, 4637, 2641, 594, 595, 596, 597, 2646, 598, 599, 600, 601, 602, 603, 604, 2654, 605, 606, 607, 2672, 2686, 2687, 2702, 2703, 2704, 2705, 2706, 2707, 2708, 2709, 2718, 782, 783, 784, 785, 811, 813, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 2874, 826, 2876, 2877, 827, 828, 829, 830, 831, 2878, 2879, 894, 2968, 2969, 957, 3023, 3024, 3025, 3026, 3027, 3028, 3029, 3030, 3031, 990, 991, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 2645, 1020, 3062, 3063, 3064, 3065, 3066, 3067, 3068, 3069, 3070, 3071, 1050, 1051, 1052, 1053, 1054, 1055, 3103, 3060, 1070, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1109, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1130, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 3199, 3230, 3231, 1200, 1201, 1202, 1203, 1204, 1205, 1206, 3282, 3286, 3287, 1240, 1241, 3288, 1243, 1244, 1245, 1246, 1247, 3289, 3290, 3291, 3292, 3293, 1264, 1265, 1266, 1267, 3353, 3354, 3355, 3356, 3357, 3358, 3359, 3386, 3390, 3391, 1362, 1363, 1364, 1365, 1366, 1367, 1368, 1369, 1370, 1371, 1372, 1373, 1374, 1375, 3420, 3451, 3452, 3453, 3454, 3455, 3504, 3505, 3506, 3507, 1471, 4364, 1487, 1488, 1489, 3615, 3624, 3625, 3626, 3627, 3628, 3629, 3630, 1597, 1598, 1599, 1656, 3725, 3726, 3729, 3730, 3731, 3732, 3733, 3734, 3735, 3736, 3737, 3738, 3739, 3740, 3741, 3742, 3743, 3760, 3761, 3762, 3763, 3764, 1718, 1719, 1720, 1721, 1722, 1723, 1724, 1725, 1726, 1727, 3775, 3792, 3793, 3794, 3795, 3797, 3798, 3799, 3800, 3801, 3802, 3803, 3804, 3805, 3806, 3807, 3824, 3825, 3826, 3827, 1780, 1781, 1782, 3828, 1790, 3863, 3864, 3865, 3866, 3867, 3868, 3869, 3870, 3871, 1846, 1847, 1848, 1849, 1850, 1851, 1852, 1853, 3898, 3899, 3900, 3901, 3902, 3903, 1880, 1882, 1883, 1884, 1885, 1886, 1887, 1904, 1905, 1906, 1907, 1908, 1909, 1910, 1911, 1912, 1913, 1941, 1942, 1944, 1945, 1946, 1947, 1948, 1949, 1950, 1951, 1969, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 2043, 2044, 2045, 2046, 2047]
    
    task_monitor = TaskMonitor(cluster=args.cluster, num_processor=args.num_num_processor, num_cycle=args.num_cycle, run_script=args.run_script, py_module=args.py_module, indices=indices, main_script_args=args.main_script_args)
    task_monitor.monitor()
