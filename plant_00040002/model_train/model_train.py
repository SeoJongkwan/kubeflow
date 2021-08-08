import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import configparser
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import h2o
from h2o.automl import H2OAutoML
import data_preprocessing #preprocessing class

import warnings
warnings.simplefilter("ignore")
pd.set_option('mode.chained_assignment', None)
plt.style.use('dark_background')


parser = argparse.ArgumentParser()
args = parser.parse_args("")

data = data_preprocessing.cleansing()
plant = data.feature_engineering()

conf = configparser.ConfigParser()
conf.read('info.init')

#=======================================CONFIGURATION=======================================*
sid = conf.get('plant', 'sid')
rtu_id_inv = conf.get('plant', 'rtu_id_inv')
feature = json.loads(conf.get('plant', 'feature'))
# args.model_path = 'ION/PV/topinfra/plant/'
# args.path = '/content/drive/MyDrive/'
#============================================================================================*

h2o.init()


train = plant[0:int(len(plant) * .7)]
test = plant[int(len(plant) * .7)+1:]

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.fit_transform(test)

train_df = pd.DataFrame(train_scaled, columns=feature)
test_df = pd.DataFrame(test_scaled, columns=feature)

htrain = h2o.H2OFrame(train_df)
htest = h2o.H2OFrame(test_df)

x = htrain.columns
y = 'dc_p'
# x.remove(y)

aml = H2OAutoML(max_runtime_secs = 600) #nfolds=9
# aml.train(x=x, y=y, training_frame=htrain, validation_frame=hval, leaderboard_frame = htest)
aml.train(x=x, y=y, training_frame=htrain)

lb = aml.leaderboard
print(lb.head())
print(aml.leader.params.keys())

pred=aml.leader.predict(htest)
pred_df = pred.as_data_frame()
htest_df = htest.as_data_frame()

def inverse_value(value):
    _value = np.zeros(shape=(len(value), len(feature)))
    _value[:,0] = value[:,0]
    inv_value = scaler.inverse_transform(_value)[:,0]
    return inv_value

inv_pred = inverse_value(pred_df['predict'].values.reshape(-1, 1))
inv_test = inverse_value(htest_df['dc_p'].values.reshape(-1, 1))

def percentage_error(actual, predicted):
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res


def evaluate_model(actual, pred):
    MAPE = "%.3f"%np.mean(np.abs(percentage_error(np.asarray(actual), np.asarray(pred))))
    MAE = "%.3f"% mean_absolute_error(actual, pred)
    RMSE = "%.3f"%np.sqrt(mean_squared_error(actual, pred))
    RSQUARED = '%.3f'%r2_score(actual, pred)
    result = '{}_Model_Evaluate_Score.json'.format(rtu_id_inv)
    print('MAPE:{} / RMSE:{}/ MAE:{} / R-Squared:{}'.format(MAPE, RMSE, MAE, RSQUARED))

    Evaluation = {}
    Evaluation['MAE'] = MAE
    Evaluation['MAPE'] = MAPE
    Evaluation['RMSE'] = RMSE
    Evaluation['R-SQUARED'] = RSQUARED
    with open(result, 'w') as f:
        json.dump(Evaluation, f, sort_keys=True)

    plt.figure(figsize=(20, 6))
    plt.plot(actual, 'limegreen', label='actual')
    plt.plot(pred, 'yellow', label='predict')
    plt.title('h2o AutoML Predict Result'); plt.xlabel('timestep'); plt.ylabel('dc_p');plt.legend()
    plt.tight_layout()
    plt.show()

evaluate_model(inv_test, inv_pred)

aml.leader.model_performance(htest)

model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:,0])
print(model_ids)
with open('{}_Algorithms.json'.format(rtu_id_inv), 'w') as f:
  f.write(str(model_ids))


best_model = h2o.download_model(aml.leader, '{}'.format(rtu_id_inv))
print(best_model)
# with open("{}_features.json".format(model_ids[0]), "w") as j:
#   json.dump(features, j)

best_algoritms = model_ids[0]
print(best_algoritms)




