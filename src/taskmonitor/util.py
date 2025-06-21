import glob
import os
import pathlib

LOG_DIR="/scratch/general/vast/u1472216/research/datastore/slurmlog"
TRACKER_DIR="/scratch/general/vast/u1472216/research/datastore/tracker"


class ProcessTable:
    def __init__(self):
        self.data = {}
        self.id = 0

    def add_process(self, process, msgQ):
        self.data[self.id] = {'process': process, 'msgQ': msgQ, 'jobid': -1, 'state': 'live'}
        self.id += 1

    def get_process_data(self, id_):
        return self.data[id_]['process'], self.data[id_]['msgQ'], self.data[id_]['jobid']
    
    def add_jobid(self, id_, jobid):
        self.data[id_]['jobid'] = jobid

    def kill_process(self, id_):
        self.data[id_]['state'] = 'dead'
        
    def get_all_ids(self):
        return list(self.data.keys())

    def get_all_live_ids(self):
        return list(filter(lambda k: self.data[k]['state'] == 'live', self.data.keys()))

    def __repr__(self):
        return ' | '.join([f"{k}:{self.data[k]['jobid']}:{self.data[k]['state']}" for k in range(self.id)])


class TableEntry:
    def __init__(self):
        self._dict = {}

    def get(self, key):
        return self._dict.get(key)

    def set(self, key, value):
        self._dict[key] = value

    def remove(self, key):
        if key in self._dict:
            del self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def __len__(self):
        return len(self._dict)

    def __str__(self):
        return str(self._dict)


def str_to_list(input_string):
    """
    """
    return list(map(int, input_string.split(',')))


def list_to_str(input_list):
    """
    """
    return ','.join(map(str, input_list))


def clear_console():
    """
    """
    os.system('cls' if os.name == 'nt' else 'clear')


def search_tracker_folder(taskid, jobid):
    """
    """
    path = glob.glob(f"{TRACKER_DIR}/{taskid}-{jobid}")
    # print(path)
    if not path:
        return path

    tracker_folder_path = path[0]
    cmplt_file_paths = glob.glob(f"{tracker_folder_path}/*.cmplt")
    cmplt = [int(os.path.splitext(os.path.basename(file_path))[0]) for file_path in cmplt_file_paths]
    return cmplt


def init_tracker_file(taskid, jobid, index):
    """
    """
    tracker_folder_path = TRACKER_DIR + f'/{taskid}-{jobid}/'
    pathlib.Path(tracker_folder_path).mkdir(parents=True, exist_ok=True)
    file_path = tracker_folder_path + f'{index:04}.txt'
    with open(file_path, 'w') as f:
        pass


def write_tracker_file(taskid, jobid, index, msg):
    """
    """
    tracker_folder_path = TRACKER_DIR + f'/{taskid}-{jobid}/'
    # pathlib.Path(tracker_folder_path).mkdir(parents=True, exist_ok=True)
    file_path = tracker_folder_path + f'{index:04}.txt'
    with open(file_path, 'a') as f:
        print(msg, file=f)


def write_complete_work(taskid, jobid, index):
    """
    """
    tracker_folder_path = TRACKER_DIR + f'/{taskid}-{jobid}/'
    pathlib.Path(tracker_folder_path).mkdir(parents=True, exist_ok=True)
    file_path = tracker_folder_path + f'{index:04}.cmplt'
    with open(file_path, 'w') as f:
        pass
