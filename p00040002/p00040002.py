import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import datetime
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import h2o
from h2o.automl import H2OAutoML
import psycopg2

parser = argparse.ArgumentParser()
args = parser.parse_args("")

import warnings
warnings.simplefilter("ignore")
pd.set_option('mode.chained_assignment', None)
h2o.init()

db = psycopg2.connect(host='34.64.92.171', dbname='solar_plant', user='solar_plant', password='ionsolarplantdev', port=5432)
cursor = db.cursor()
#=======================================CONFIGURATION=======================================*
args.sid = '0004_0002'
args.rtu_id_inv = "0004_0002_I_0001"
args.features = ['dc_p', 's_ir', 'h_ir', 'm_t']
# args.model_path = 'ION/PV/topinfra/plant/'
# args.path = '/content/drive/MyDrive/'
#============================================================================================*

inv_cond = "WHERE rtu_id = '{}'".format(args.rtu_id_inv)
select = "SELECT * FROM train {}".format(inv_cond)

cursor.execute(select)
plant = pd.DataFrame(cursor.fetchall())
plant.columns = [desc[0] for desc in cursor.description]
# plant.to_csv("{}/{}/plant.csv".format(args.sid, args.rtu_id_inv))

# plant = pd.read_csv('{}/{}/plant.csv'.format(args.sid, args.rtu_id_inv), index_col=0)

features = ['dc_p', 's_ir', 'h_ir', 'm_t'] #'a_t'

plant_feature = plant[features]
plant_feature.head()

train = plant_feature[0:int(len(plant_feature) * .7)]
test = plant_feature[int(len(plant_feature) * .7)+1:]

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.fit_transform(test)

train_df = pd.DataFrame(train_scaled, columns=features)
test_df = pd.DataFrame(test_scaled, columns=features)

htrain = h2o.H2OFrame(train_df)
htest = h2o.H2OFrame(test_df)

x = htrain.columns
y = 'dc_p'
# x.remove(y)

aml = H2OAutoML(max_runtime_secs = 600) #nfolds=9
# aml.train(x=x, y=y, training_frame=htrain, validation_frame=hval, leaderboard_frame = htest) 
aml.train(x=x, y=y, training_frame=htrain) 

lb = aml.leaderboard
lb.head()

aml.leader.params.keys()


pred=aml.leader.predict(htest)
pred_df = pred.as_data_frame()
pred_df
htest_df = htest.as_data_frame()
htest_df

def inverse_value(value):
    _value = np.zeros(shape=(len(value), len(features)))
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
    
import json

plt.style.use('dark_background')

def evaluate_model(actual, pred):
    MAPE = "%.3f"%np.mean(np.abs(percentage_error(np.asarray(actual), np.asarray(pred))))
    MAE = "%.3f"% mean_absolute_error(actual, pred)
    RMSE = "%.3f"%np.sqrt(mean_squared_error(actual, pred))
    RSQUARED = '%.3f'%r2_score(actual, pred)
    result = '{}_Model_Evaluate_Score.json'.format(args.rtu_id_inv)
    print('MAPE: {} / RMSE: {}/ MAE: {}/ R-Squared: {}'.format(MAPE, RMSE, MAE, RSQUARED))

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
    # plt.savefig(env.get_models_path('result/model_{}_v{}.png'.format(plant_name, version)))
    plt.tight_layout()
    plt.show()

evaluate_model(inv_test, inv_pred)

aml.leader.model_performance(htest)


model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:,0])
model_ids
with open('{}_Algorithms.json'.format(args.rtu_id_inv), 'w') as f:
  f.write(str(model_ids))


best_model = h2o.download_model(aml.leader, '{}'.format(args.rtu_id_inv))
# with open("{}_features.json".format(model_ids[0]), "w") as j:
#   json.dump(features, j)

best_algoritms = model_ids[0]
print(best_algoritms)




