import os
from dotenv import load_dotenv
import psycopg2
from tenacity import retry, wait_random_exponential, stop_after_attempt

load_dotenv('.env')

paramz = dict()

paramz["localhost"] = {
    "dbname": os.getenv("DBNAME"),
    "user": os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASSWORD"),
    "host": os.getenv("PG_SERVER", "localhost"),
    "port": os.getenv("PG_PORT", 5432),
}


class spsql:
    def __init__(self, param="localhost", autocommit=True):
        self.autocommit = autocommit
        if type(param) == str:
            self.param = paramz[param]
        elif type(param) == dict:
            self.param = param
        self.connect()

    def connect(self):
        conn = psycopg2.connect(**self.param)
        conn.autocommit = self.autocommit
        self._conn = conn
        self._curs = conn.cursor()
        return True

    def _getcurs(self):
        if self._curs.rownumber < self._curs.rowcount or (
            self._curs.rownumber == 0 and self._curs.rowcount == 0
        ):
            return self._curs
        else:
            self._curs.close()
            self._curs = self.conn.cursor()
            return self._curs

    def _getconn(self):
        if not self.connected():
            self.connect()
            raise Warning("DB CONNECTION RECOVERED")
        if not self.connected():
            raise Warning("CANT RECONNECT")
        return self._conn

    def connected(self):
        return self._conn.closed == 0

    curs = property(fget=_getcurs)
    conn = property(fget=_getconn)


def getconn(autocommit=True):
    conn = psycopg2.connect(**param)
    conn.autocommit = autocommit
    return conn
