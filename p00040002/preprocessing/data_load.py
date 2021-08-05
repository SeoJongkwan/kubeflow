import pandas as pd
import psycopg2
import argparse
import configparser
import json

parser = argparse.ArgumentParser()
args = parser.parse_args("")

conf = configparser.ConfigParser()
conf.read('info.init')

host = conf.get('DB', 'host')
dbname = conf.get('DB', 'dbname')
user = conf.get('DB', 'user')
password = conf.get('DB', 'password')
port = conf.get('DB', 'port')
table = json.loads(conf.get('DB','table')) #train

sid = conf.get('plant', 'sid')
rtu_id_inv = conf.get('plant', 'rtu_id_inv')

class load:
    def __init__(self):
        self.con = psycopg2.connect(host=host, dbname=dbname, user=user, password=password, port=port)
        self.cursor = self.con.cursor()

    def select(self):
        cond = "rtu_id = '{}'".format(rtu_id_inv)
        query = "SELECT * FROM {} WHERE {}".format(table[2], cond)

        self.cursor.execute(query)
        plant = pd.DataFrame(self.cursor.fetchall())
        plant.columns = [desc[0] for desc in self.cursor.description]
        plant.to_csv("{}_plant.csv".format(rtu_id_inv))

        return plant

# plant = load()
# data = plant.select()