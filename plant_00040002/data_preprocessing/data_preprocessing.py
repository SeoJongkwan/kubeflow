import pandas as pd
import json
import argparse
import configparser

parser = argparse.ArgumentParser()
args = parser.parse_args("")

conf = configparser.ConfigParser()
conf.read('info.init')

args.sid = conf.get('plant', 'sid')
args.rtu_id_inv = conf.get('plant', 'rtu_id_inv')
args.feature = json.loads(conf.get('plant', 'feature'))
print("sid:", args.sid + "\nrtu_id_inv:", args.rtu_id_inv + "\n")
print("features: {}\n".format(args.feature))

PVC = "pvmodel-vol-1/"
data_path = PVC + "data/" + args.sid
model_path = PVC + "model/{}/{}".format(args.sid, args.rtu_id_inv)
original_data = "{}_original.csv".format(args.rtu_id_inv)
cleaning_data = "{}_Preprocessing.csv".format(args.rtu_id_inv)

load_data = pd.read_csv(data_path + '/' + original_data, index_col=[0])

def check_missing_value(data, col):
    print('Check Count of Missing Value on Each Column:\n{}'.format(data.isnull().sum()))
    if data[col].isnull().sum() != 0:
        nan_row = data[data[col].isnull()]
        print("Remove Index\n", nan_row.index)
        data = data.drop(nan_row.index).reset_index(drop=True)
        print('Drop Missing Value Result:\n{}'.format(data.isnull().sum()))
    return data

def feature_engineering(data):
    plant_feature = data[args.feature]
    plant = check_missing_value(plant_feature, 'dc_p')
    print("Persistent Volume Claims:", PVC)
    print("File Saved:", cleaning_data)
    print("Volume Container Path:", data_path)
    plant.to_csv('{}/{}'.format(data_path, cleaning_data))
    return plant

feature_engineering(load_data)
