import pandas as pd
import json
import argparse
import configparser

parser = argparse.ArgumentParser()
args = parser.parse_args("")

conf = configparser.ConfigParser()
conf.read('info.init')

rtu_id_inv = conf.get('plant', 'rtu_id_inv')

args.volume_mount_path = "/data/"

load_data = pd.read_csv(args.volume_mount_path + "{}_plant.csv".format(rtu_id_inv), index_col=[0])

def check_missing_value(feature_data):
    print('Check Count of Missing Value on Each Column:\n{}'.format(feature_data.isnull().sum()))
    for i in range(len(feature_data.isnull().sum())):
        if feature_data.isnull().sum()[i] != 0:
            feature_data = feature_data.dropna(axis=0)
            print('Drop Missing Value Result:\n{}'.format(feature_data.isnull().sum()))
    return feature_data

def feature_engineering(data):
    feature = json.loads(conf.get('plant', 'feature'))
    print("features: {}".format(feature))
    plant_feature = data[feature]
    plant = check_missing_value(plant_feature)
    print("Preprocessing Finished")
    plant.to_csv(args.volume_mount_path + "{}_preprocessing.csv".format(rtu_id_inv))
    return plant

feature_engineering(load_data)
