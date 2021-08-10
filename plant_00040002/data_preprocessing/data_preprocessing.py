# import json
# import argparse
# import configparser
#
# parser = argparse.ArgumentParser()
# args = parser.parse_args("")
#
# conf = configparser.ConfigParser()
# conf.read('info.init')
#
# rtu_id_inv = conf.get('plant', 'rtu_id_inv')
#
# args.data_path = "/data/"
#
# def data_loader():
#     df = pd.read_csv("/data/titanic_train.csv")
#     print(df.head(1))
#
#
# def check_missing_value(feature_data):
#     print('Check Count of Missing Value on Each Column:\n{}'.format(feature_data.isnull().sum()))
#     for i in range(len(feature_data.isnull().sum())):
#         if feature_data.isnull().sum()[i] != 0:
#             feature_data = feature_data.dropna(axis=0)
#             print('Drop Missing Value Result:\n{}'.format(feature_data.isnull().sum()))
#     return feature_data
#
# def feature_engineering(data):
#     feature = json.loads(conf.get('plant', 'feature'))
#     print("features: {}".format(feature))
#     plant_feature = data[feature]
#     plant = check_missing_value(plant_feature)
#     plant.to_csv(args.data_path + "{}_plant.csv".format(rtu_id_inv))
#     return plant

import glob
import pandas as pd
import tarfile
import urllib.request


def download_and_merge_csv(url: str, output_csv: str):
    with urllib.request.urlopen(url) as res:
        tarfile.open(fileobj=res, mode="r|gz").extractall('data')
    df = pd.concat(
        [pd.read_csv(csv_file, header=None)
         for csv_file in glob.glob('data/*.csv')])
    df.to_csv(output_csv, index=False, header=False)

download_and_merge_csv(
    url='minio://mlpipeline/artifacts/data-load-pipeline-bj7d4/data-load-pipeline-bj7d4-4280271310/data-load-container-load_data.tgz',
    output_csv='merged_data.csv')