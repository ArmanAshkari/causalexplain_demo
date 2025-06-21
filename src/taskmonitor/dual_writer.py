import sys

class DualWriter:
    def __init__(self, file_path, file_only=False):
        self.console = sys.__stdout__  # Save the reference to the original console output
        self.file = open(file_path, 'a')  # Open the file in append mode
        self.file_only = file_only

    def write(self, message):
        # Write to the console
        if not self.file_only:
            self.console.write(message)
        # Write to the file
        self.file.write(message)

    def flush(self):
        # Flush both stdout and the file buffer
        if not self.file_only:
            self.console.flush()
        self.file.flush()

    def __del__(self):
        self.file.close()