from datetime import datetime
import re
import subprocess
import time
from collections import defaultdict
from .slurm_util import cancel_job


class JobInfo:
    def __init__(self, jobid, state):
        self.jobid = jobid
        self.state = state

    def __repr__(self):
        return (f"JobInfo(jobid={self.jobid}, state={self.state}")
    

class JobMonitor:
    R_THRSHLD = 120
    PD_THRSHLD = 120
    SB_THRSHLD = 120
    
    def __init__(self, taskid, jobid, cluster, job_arr_size, jobid_indices_map, total, msgQ):
        self.taskid = taskid
        self.jobid = jobid
        self.cluster = cluster
        self.job_arr_size = job_arr_size 
        self.jobid_indices_map = jobid_indices_map 
        self.total = total
        self.msgQ = msgQ
        self.job_states = {f'{self.jobid}_{i}': 'SB' for i in range(self.job_arr_size)}
        self.RTs_to_report = set()
        self.reported_RTs = set()
        self.R_counter = defaultdict(int)
        self.PD_counter = defaultdict(int)
        self.SB_counter = defaultdict(int)

    
    @staticmethod
    def get_jobinfos(cluster):
        """
        Gets (jobid, state) of jobs of current user
        """
        try:
            command = f'squeue --format="%i %t" -u $USER -M {cluster}'
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            lines = result.stdout.splitlines()   
            return list(map(lambda line: JobInfo(*line.split()), lines[1:]))
        except Exception as e:
            return []

    
    @staticmethod
    def expand_range(jobids):
        """
        Expand jobid_[a:b] to [jobid_a, ... , jobid_b]
        Can also handle jobid_[a:b%c], jobid_[a], jobid_[a%c]
        Cases liks jobid_a are unchanged
        """
        expanded_list = []
        pattern = r'(\d+_)\[(\d+)(?:-(\d+))?(?:%[^\]]+)?\]'
        for jobid in jobids:
            match = re.match(pattern, jobid)
            if match:
                base = match.group(1)  # '1976260_'
                start = int(match.group(2))  # Start of the range or single number
                end = int(match.group(3)) if match.group(3) else start  # End of range or same as start for single number
                # Append each number in the range or the single number
                expanded_list.extend([f'{base}{i}' for i in range(start, end + 1)])
            else:
                expanded_list.append(jobid)    
        
        return expanded_list

    
    def filter_jobid_by_status(self):
        """
        """
        jobinfo_list = JobMonitor.get_jobinfos(self.cluster)

        in_jobinfo = list(filter(lambda x: x.jobid.startswith(self.jobid) and x.state in ['R', 'S', 'ST'], jobinfo_list))
        in_jobids = list(map(lambda x: x.jobid, in_jobinfo))

        pending_jobinfo = list(filter(lambda x: x.jobid.startswith(self.jobid) and x.state in ['PD'], jobinfo_list))
        pending_jobids = JobMonitor.expand_range(list(map(lambda x: x.jobid, pending_jobinfo)))

        out_jobinfo = list(filter(lambda x: x.jobid.startswith(self.jobid) and x.state in ['PR', 'CA', 'CG','CD', 'F', 'NF', 'DL', 'TO'], jobinfo_list))
        out_jobids = list(map(lambda x: x.jobid, out_jobinfo))
        
        return in_jobids, pending_jobids, out_jobids

    
    def update_job_states(self, in_jobids, pending_jobids, out_jobids):
        """
        """
        next_job_states = {}
        for k, v in self.job_states.items():
            if v == 'SB':
                if k in out_jobids:
                    next_job_states[k] = 'RT'
                    self.RTs_to_report.add(k)
                
                elif k in in_jobids:
                    next_job_states[k] = 'R'
                
                elif k in pending_jobids:
                    next_job_states[k] = 'PD'
                
                else:
                    if self.SB_counter[k] > JobMonitor.SB_THRSHLD:
                        next_job_states[k] = 'RT'
                        self.RTs_to_report.add(k)
                        cancel_job(k)
                    else:
                        next_job_states[k] = 'SB'
                        self.SB_counter[k] += 1
            
            elif v == 'PD':
                if k in out_jobids:
                    next_job_states[k] = 'RT'
                    self.RTs_to_report.add(k)
                
                elif k in in_jobids:
                    next_job_states[k] = 'R'
                
                elif k in pending_jobids:
                    next_job_states[k] = 'PD'
                    self.PD_counter[k] = 0
                
                else:
                    if self.PD_counter[k] > JobMonitor.PD_THRSHLD:
                        next_job_states[k] = 'RT'
                        self.RTs_to_report.add(k)
                        cancel_job(k)
                    else:
                        next_job_states[k] = 'PD'
                        self.PD_counter[k] += 1
            
            elif v == 'R':
                if k in out_jobids:
                    next_job_states[k] = 'RT'
                    self.RTs_to_report.add(k)
                
                elif k in in_jobids:
                    next_job_states[k] = 'R'
                    self.R_counter[k] = 0
                
                else:
                    if self.R_counter[k] > JobMonitor.R_THRSHLD:
                        next_job_states[k] = 'RT'
                        self.RTs_to_report.add(k)
                        cancel_job(k)
                    else:
                        next_job_states[k] = 'R'
                        self.R_counter[k] += 1
            
            elif v == 'RT':
                next_job_states[k] = 'RT'
        
        self.job_states = next_job_states

    
    def track_job_arr(self):
        """
        """
        while True:
            time.sleep(5)
            
            in_jobids, pending_jobids, out_jobids = self.filter_jobid_by_status()
            self.update_job_states(in_jobids, pending_jobids, out_jobids)
            
            #region
            # print('\n',
            #       datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'\n',
            #       'Jobid', self.jobid, '\n',
            #       'Job queue', in_jobids, pending_jobids, out_jobids, '\n',
            #       'Job states', self.job_states, '\n',
            #       'SB_counter', self.SB_counter, '\n',
            #       'PD_counter', self.PD_counter, '\n',
            #       'R_counter', self.R_counter, '\n',
            #       '\n')
            #endregion

            self.report_RTs()

            if not in_jobids and not pending_jobids \
                and all(v == 'RT' for v in self.job_states.values()):
                    # print(f'\n***Track job array returning from {self.jobid}***\n')     
                    return 

    
    def report_RTs(self):
        """
        """
        if self.RTs_to_report:
            self.msgQ.put(('REPORT', self.RTs_to_report))
            assert len(self.RTs_to_report & self.reported_RTs) == 0, 'Double reported'
            self.reported_RTs.update(self.RTs_to_report)
            self.RTs_to_report = set()
