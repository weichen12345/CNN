# coding=utf-8
import mysql.connector


mysql_configs = {
    '172.16.0.76': {
        'host': '172.16.0.76',
        'user': 'likai',
        'password': '1qaz@WSX3edc',
        'port': 3306,
        'database': 'law',
        'charset': 'utf8'
        },
    '172.16.0.20': {
        'host': '172.16.0.20',
        'user': 'zhangxiaogang',
        'password': 'gangxiaozhang',
        'port': 3306,
        'database': 'court_notice',
        'charset': 'utf8'
              },
    '172.16.0.11': {
        'host': '172.16.0.11',
        'user': 'bigdata1',
        'password': 'aaBigDataZZ123$',
        'port': 3306,
        'database': 'ljzxdb',
        'charset': 'utf8mb4'
              }
                 }


class MySQL(object):

    fetch_batch_size = 1000

    def __init__(self, host, db=None, auto_commit=True):
        if db:
            mysql_configs['database'] = db
        self.connection = mysql.connector.connect(**mysql_configs[host])
        self.connection.autocommit = auto_commit
        self.cursor = self.connection.cursor()

    def execute_query(self, sql, params=()):
        self.connection.ping(True)
        self.cursor.execute(sql, params)
        while True:
            _res = self.cursor.fetchmany(self.fetch_batch_size)
            if _res:
                for _row in _res:
                    yield _row
            else:
                break

    def execute_update(self, sql, params=()):
        self.connection.ping(True)
        self.cursor.execute(sql, params)

    def execute_many_update(self, sql, params):
        self.connection.ping(True)
        self.cursor.executemany(sql, params)


if __name__ == '__main__':
    mysql_76 = MySQL('172.16.0.76')
    mysql_76.fetch_batch_size = 1
    res = mysql_76.execute_query("""
    select '123'
    union all
    select '456'
    union all
    select '789'
    union all
    select '012'
    union all
    select '023'
    """)
    for row in res:
        print(row)


