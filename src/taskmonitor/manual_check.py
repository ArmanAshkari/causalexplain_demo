import glob
import os
from .util import TRACKER_DIR, search_tracker_folder
import itertools


def status_check(taskid):
    """
    """
    path = glob.glob(f"{TRACKER_DIR}/{taskid}-*")
    if not path:
        return

    jobids = list(map(lambda x: os.path.basename(x).split('-')[1], path))

    intersections = []
    jobid_combinations = []

    i = 1
    while True:
        comb_jobid_len_i = list(itertools.combinations(jobids, i))
        intersect_len_i = []

        for comb in comb_jobid_len_i:
            # Get the set of files in those i folders and find intersections among them.
            list_of_set_of_files = list(map(lambda c: set(search_tracker_folder(taskid, c)), comb))
            intersect_len_i.append(set.intersection(*list_of_set_of_files))

        # There's no intersection of len i. So, len i+1 intersections are not possible. Break.
        if all(list(map(lambda x: len(x) == 0, intersect_len_i))):
            break
        
        # Save the intersection and jobid_combination in descending order of length.
        intersections = intersect_len_i + intersections
        jobid_combinations = comb_jobid_len_i + jobid_combinations
        i += 1

    done = set()
    for comb, intersect in zip(jobid_combinations, intersections):
        if not intersect:
            continue
        not_repeated = intersect - done
        if not not_repeated:
            continue
        # print(f'{sorted(not_repeated)}:{len(not_repeated)} found in {sorted(comb)}', f'{"Multiple instance!" if len(comb) > 1 else ""}')
        print(f'{len(not_repeated)} files found in {sorted(comb)}', f'{"Multiple instance!" if len(comb) > 1 else ""}')
        done.update(not_repeated)

    print(f'Taskid: {taskid}')
    print(f'Jobids: {jobids}')
    print(f'Total found: {len(done)}')


if __name__=="__main__":
    status_check(taskid='165')