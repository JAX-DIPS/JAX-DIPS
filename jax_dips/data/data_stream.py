import itertools
import os
import pickle
import queue
import sqlite3
import threading
import time
import uuid


class StateData:
    """
    To manage data streamed from state and store offline.
    """

    def __init__(self, q, content_dir=None, cols=["t", "U", "kappaM", "nx", "ny", "nz"]) -> None:
        self._q = q
        self._cols = set(cols)
        self._listen = None

        self._run_id = str(uuid.uuid1())

        if content_dir is None:
            content_dir = "/tmp/dips"
        self._content_dir = os.path.join(content_dir, self._run_id)
        self._db_url = os.path.join(self._content_dir, "state.db")
        self._state_dir = os.path.join(self._content_dir, "state")

        os.makedirs(self._state_dir, exist_ok=True)

        self._initialize_db()

    def _initialize_db(self):
        """
        Create a new database and initialize the table.
        """
        conn = sqlite3.connect(self._db_url, uri=True, check_same_thread=False)
        ddl_stmt = f"""
                    CREATE TABLE IF NOT EXISTS state_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        {' TEXT, '.join(self._cols)} TEXT
                    );
                    """
        conn.executescript(ddl_stmt)

    def __len__(self):
        conn = sqlite3.connect(self._db_url, uri=True, check_same_thread=False)
        res = conn.execute("SELECT count(*) FROM state_log")
        return res.fetchone()[0]

    def _start(self):
        """
        Listen for incoming data and write it to the database.
        """
        conn = sqlite3.connect(self._db_url, uri=True, check_same_thread=False)
        insert_stmt = f"""
             INSERT INTO state_log({', '.join(self._cols)})
             VALUES({', '.join(list(itertools.repeat('?', len(self._cols))))})
             """
        while True:
            try:
                data = self._q.get()
                t = float(data["t"])
                if t == -1 and not self._listen:
                    break
                rec_path = os.path.join(self._state_dir, str(t))
                os.mkdir(rec_path)
                locs = {}
                for k in self._cols:
                    d = os.path.join(rec_path, f"{k}.pkl")
                    f = open(d, "wb")
                    pickle.dump(data[k], f)
                    locs[k] = d

                conn.execute(insert_stmt, [locs[k] for k in self._cols])
                conn.commit()
                self._q.task_done()
            except queue.Empty:
                if not self._listen:
                    break
            except Exception as e:
                print(e)
        conn.close()

    def keys(self):
        return self._cols

    def get(self, key, index):
        conn = sqlite3.connect(self._db_url, uri=True, check_same_thread=False)
        try:
            res = conn.execute(f"SELECT {key} FROM state_log WHERE id = ?", [index + 1])
            return pickle.load(open(res.fetchone()[0], "rb"))
        except Exception as e:
            print(e)
        finally:
            conn.close()

    def start(self):
        self._worker = threading.Thread(target=self._start)
        print("Starting state msg listener...")
        self._worker.start()

    def queue_state(self, state):
        self._q.put(state)

    def stop(self):
        self._listen = False
        while not self._q.empty():
            time.sleep(1)
        self.queue_state({"t": -1})
