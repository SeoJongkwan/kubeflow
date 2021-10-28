import pandas as pd
import psycopg2
import argparse
import configparser
import json

parser = argparse.ArgumentParser()
args = parser.parse_args("")

conf = configparser.ConfigParser()
conf.read('info.init')

dbname = conf.get('DB', 'dbname')
host = conf.get('DB', 'host')
user = conf.get('DB', 'user')
password = conf.get('DB', 'password')
port = conf.get('DB', 'port')
table = json.loads(conf.get('DB','table')) #train table
print("<DB Info>")
print("dbname:", dbname + "\nhost:", host + "\nport:", port)

args.sid = conf.get('plant', 'sid')
args.rtu_id_inv = conf.get('plant', 'rtu_id_inv')
print("<Plant Info>")
print("sid:", args.sid + "\nrtu_id_inv:", args.rtu_id_inv + "\n")

PVC = "pvmodel-vol-1/"
data_path = PVC + "data/" + args.sid
model_path = PVC + "model/{}/{}".format(args.sid, args.rtu_id_inv)
original_data = "{}_original.csv".format(args.rtu_id_inv)

con = psycopg2.connect(host=host, dbname=dbname, user=user, password=password, port=port)
cursor = con.cursor()

cond = "sid = '{}'".format(args.sid)
query = "SELECT * FROM {} WHERE {}".format(table[2], cond)

cursor.execute(query)
plant = pd.DataFrame(cursor.fetchall())
plant.columns = [desc[0] for desc in cursor.description]
plant.to_csv('{}/{}'.format(data_path, original_data))

print("Persistent Volume Claims:", PVC)
print("File Saved:", original_data)
print("Volume Container Path:", data_path)
