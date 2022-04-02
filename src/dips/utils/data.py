import pdb
import itertools
import queue
import sqlite3

from contextlib import closing
import threading
import pickle

import time

class StateData:

    def __init__(self,
                 q,
                 db_url='/tmp/dips.db',
                 cols = ['t', 'U', 'kappaM', 'nx', 'ny', 'nz']) -> None:
        self._db_url = db_url
        self._q = q
        self._cols = set(cols)

        self._listen = None

        self._initialize_db()

    def _initialize_db(self):
        '''
        Create a new database and initialize the table.
        '''
        conn = sqlite3.connect(self._db_url,
                                     uri=True,
                                     check_same_thread=False)
        ddl_stmt = f'''
                    CREATE TABLE IF NOT EXISTS state_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        {' TEXT, '.join(self._cols)} TEXT
                    );
                    '''
        conn.executescript(ddl_stmt)

    def __len__(self):
        conn = sqlite3.connect(self._db_url,
                                uri=True,
                                check_same_thread=False)
        res = conn.execute('SELECT count(*) FROM state_log')
        return res.fetchone()[0]

    def _start(self):
        '''
        Listen for incoming data and write it to the database.
        '''
        conn = sqlite3.connect(self._db_url,
                                uri=True,
                                check_same_thread=False)
        insert_stmt =\
            f'''
            INSERT INTO state_log({', '.join(self._cols)})
            VALUES({', '.join(list(itertools.repeat('?', len(self._cols))))})
            '''
        while True:
            try:
                data = self._q.get()
                conn.execute(
                   insert_stmt,
                   [pickle.dumps(data[k]) for k in self._cols])
                conn.commit()
            except queue.Empty:
                if not self._listen:
                    break
            except Exception as e:
                print(e)
        conn.close()

    def keys(self):
        return self._cols

    def get(self, key, index):
        conn = sqlite3.connect(self._db_url,
                                uri=True,
                                check_same_thread=False)
        # print(f'SELECT {key} FROM state_log WHERE ID = ?', index)
        res = conn.execute(f'SELECT {key} FROM state_log WHERE ID = ?', [index + 1])
        return pickle.loads(res.fetchone()[0])

    def start(self):
        worker = threading.Thread(target=self._start)
        print('Starting state msg listener...')
        worker.start()

    def queue_state(self, state):
        self._q.put(state)

    def stop(self):
        self._listen = False
        while not self._q.empty():
            time.sleep(1)