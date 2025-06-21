from pathlib import Path
from shutil import rmtree
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait
from src.util.directories import SCRATCH_DIR

import os


# Function to delete a subdirectory
def delete_subdirectory(subdir):
    try:
        if subdir.is_dir():
            rmtree(subdir)
            return f"Deleted: {subdir}"
        else:
            return f"Skipped (Not a directory): {subdir}"
    except Exception as e:
        return f"Error deleting {subdir}: {e}"


def parallel_delete_subdirectories(parent_dir, num_processor):
    """
    """
    parent_dir = Path(parent_dir)
    subdirs = [d for d in parent_dir.iterdir() if d.is_dir()]
    with ProcessPoolExecutor(num_processor) as executor:
        futures = [executor.submit(delete_subdirectory, d) for d in subdirs]
        wait(futures)
    
    print('All threads finished')


def delete_file(file_path):
    """Delete a single file."""
    try:
        os.remove(file_path)
        return f"Deleted: {file_path}"
    except Exception as e:
        return f"Error deleting {file_path}: {e}"

def parallel_delete(folder_path, num_processor):
    """Delete all files in a folder in parallel."""
    if not os.path.isdir(folder_path):
        print("The provided path is not a directory.")
        return
    
    # List all files in the folder
    files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
    
    if not files:
        print("No files found in the folder.")
        return
    
    # with ProcessPoolExecutor(num_processor) as executor:
    #     futures = [executor.submit(delete_file, file) for file in files]
    #     wait(futures)

    #     for result in futures:
    #         print(result)

    # Use ThreadPoolExecutor for I/O-bound tasks or ProcessPoolExecutor for CPU-bound tasks
    with ThreadPoolExecutor(max_workers=num_processor) as executor:
        results = executor.map(delete_file, files)
    
        print('All threads finished')



if __name__ == "__main__":
    """
    python -m src.util.parallel_deletion
    """
    # PARENT_DIR = Path(f'{SCRATCH_DIR}/adult/rand_forest/rf')
    PARENT_DIR = '/scratch/general/vast/u1472216/research/datastore/tracker/delete'
    # PARENT_DIR = '/scratch/general/vast/u1472216/research/datastore/tracker/delete/534-3213256'
    # PARENT_DIR ='/scratch/general/vast/u1472216/research/datastore/bbmlexplain/so/baseline_saved_models/_baseline_1/euclidean/'
    parallel_delete_subdirectories(parent_dir=PARENT_DIR, num_processor=192)
    # parallel_delete(folder_path=PARENT_DIR, num_processor=56)