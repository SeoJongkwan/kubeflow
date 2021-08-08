import pandas as pd
import psycopg2
import argparse
import configparser
import json

import kfp
import kfp.dsl as dsl
from kfp import components
import kfp.compiler as compiler

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

con = psycopg2.connect(host=host, dbname=dbname, user=user, password=password, port=port)
cursor = con.cursor()

def select():
    cond = "rtu_id = '{}'".format(rtu_id_inv)
    query = "SELECT * FROM {} WHERE {}".format(table[2], cond)

    cursor.execute(query)
    plant = pd.DataFrame(cursor.fetchall())
    plant.columns = [desc[0] for desc in cursor.description]
    print("file saved")
    plant.to_csv("{}_plant.csv".format(rtu_id_inv))

    return plant

@dsl.pipeline(
    name="data load pipeline",
    description="apply volume"
)

def pipeline():
    data_vop=dsl.VolumeOp(
        name="data-volume",
        resource_name="data-pvc",
        modes=dsl.VOLUME_MODE_RWO,
        size="1Gi"
    )

    data_op=dsl.ContainerOp(
        name="data load container",
        image="bellk/load_data:1.2",
        # command=["sh","-c"],
        # arguments=["/data/0004_0002_I_0001_plant.csv"],
        file_outputs={"load_data":"/0004_0002_I_0001_plant.csv"},
        pvolumes={"/data": data_vop.volume}
    )

# import os
# parser = argparse.ArgumentParser()
# parser.add_argument(
#     '--data_path',
#     type=str,
#     help='Path to the training data'
# )
#
# args = parser.parse_args()
# # print("===== DATA =====")
# # print("DATA PATH: " + args.data_path)
# # print("LIST FILES IN DATA PATH...")
# print(os.listdir(args.data_path))
# args

