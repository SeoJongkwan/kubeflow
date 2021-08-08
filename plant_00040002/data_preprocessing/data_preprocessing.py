import json
import argparse
import configparser
import data_load

parser = argparse.ArgumentParser()
args = parser.parse_args("")

conf = configparser.ConfigParser()
conf.read('info.init')

class cleansing:
    def __init__(self):
        self.data = data_load.load().select()

    def check_missing_value(self, feature_data):
        print('Check Count of Missing Value on Each Column:\n{}'.format(feature_data.isnull().sum()))
        for i in range(len(feature_data.isnull().sum())):
            if feature_data.isnull().sum()[i] != 0:
                feature_data = feature_data.dropna(axis=0)
                print('Drop Missing Value Result:\n{}'.format(feature_data.isnull().sum()))
        return feature_data

    def feature_engineering(self):
        feature = json.loads(conf.get('plant', 'feature'))
        print("features: {}".format(feature))
        # plant = pd.read_csv('{}_plant.csv'.format(rtu_id_inv), index_col=0)
        plant_feature = self.data[feature]
        plant = cleansing().check_missing_value(plant_feature)

        return plant

