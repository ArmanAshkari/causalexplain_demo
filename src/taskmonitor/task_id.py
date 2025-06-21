import sqlite3
from .util import TRACKER_DIR

class SQLiteDatabaseIndexCounter:
    def __init__(self, db_path=f"{TRACKER_DIR}/index_counter.db", table_name="counter", start=1):
        """
        Initialize the counter with a start value and connect to the SQLite database.
        :param db_path: Path to the SQLite database file.
        :param table_name: The name of the table to store the index.
        :param start: The value at which the counter starts (default is 1).
        """
        self.db_path = db_path
        self.table_name = table_name
        self._start = start
        self._initialize_database()

    def _initialize_database(self):
        """
        Initialize the database and create the counter table if it doesn't exist.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Create table if not exists
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id INTEGER PRIMARY KEY
                )
            """)
            # Check if there is an existing index
            cursor.execute(f"SELECT id FROM {self.table_name} LIMIT 1")
            row = cursor.fetchone()
            if row is None:
                # Insert the start value if no index exists
                cursor.execute(f"INSERT INTO {self.table_name} (id) VALUES (?)", (self._start,))
                conn.commit()

    def _get_current_index(self):
        """
        Retrieve the current index from the database.
        :return: The current index value.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT id FROM {self.table_name} LIMIT 1")
            return cursor.fetchone()[0]

    def _set_index(self, new_value):
        """
        Set the current index in the database.
        :param new_value: The new index value to set.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"UPDATE {self.table_name} SET id = ? WHERE id = ?", (new_value, self._get_current_index()))
            conn.commit()

    def get_next_index(self):
        """
        Increment the counter in the database and return the next index.
        :return: The next index value.
        """
        current_index = self._get_current_index()
        self._set_index(current_index + 1)
        return current_index

    def reset(self):
        """
        Reset the counter to the initial start value and update the database.
        """
        self._set_index(self._start)

    def set_index(self, new_value):
        """
        Manually set the current index value in the database.
        :param new_value: The new index value to set.
        """
        self._set_index(new_value)

    def get_current_index(self):
        """
        Return the current index without incrementing.
        :return: The current index value.
        """
        return self._get_current_index()

    def __repr__(self):
        return f"SQLiteDatabaseIndexCounter(current_index={self.get_current_index()}, start={self._start})"


# # Example usage:
# counter = SQLiteDatabaseIndexCounter()
# print(counter.get_next_index())  # Output: Next index
# print(counter.get_current_index())  # Output: Current index without incrementing
# # counter.set_index(100)  # Manually set index to 100
# print(counter.get_next_index())  # Output: 100
# # counter.reset()  # Reset the counter