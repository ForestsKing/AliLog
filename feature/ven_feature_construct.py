import joblib
import numpy as np
import pandas as pd


def getLog(log, sn, fault_time):
    tmp = log[(log['sn'] == sn)].copy(deep=True)
    tmp['delta'] = tmp['fault_time'].apply(lambda x: abs((fault_time - x)))
    if len(tmp) > 0:
        tmp = tmp[tmp['delta'] < pd.to_timedelta("1D")]
        if len(tmp) > 0:
            tmp['delta'] = tmp['fault_time'].apply(lambda x: (fault_time - x).total_seconds())
            tmp = tmp.sort_values('delta')
            return [tmp['delta'].values[0], tmp['module_cause'].values[0], tmp['module'].values[0]]
        else:
            return [0, "", ""]
    else:
        return [0, "", ""]


def VenFeatureConstruct(log, submit, train=True):
    data = submit.copy(deep=True)
    log['fault_time'] = pd.to_datetime(log['fault_time'])
    data['fault_time'] = pd.to_datetime(data['fault_time'])

    data['log'] = data.apply(lambda x: getLog(log, x['sn'], x['fault_time']), axis=1)

    data['VEN_delta'] = data['log'].apply(lambda x: x[0])
    data['module_cause'] = data['log'].apply(lambda x: x[1])
    data['module'] = data['log'].apply(lambda x: x[2].split(','))

    if train:
        module2id = {}
        for i, module in enumerate(sorted(list(set([i for item in data['module'].values.tolist() for i in item])))):
            module2id[module] = i
        joblib.dump(module2id, './user_data/model_data/ven_module2id.pkl')
    else:
        module2id = joblib.load('./user_data/model_data/ven_module2id.pkl')

    data['module'] = data['module'].apply(lambda x: list(map(lambda a: module2id.get(a, len(module2id)), x)))
    data['module'] = data['module'].apply(lambda x: (np.eye(len(module2id) + 1)[x]).max(axis=0))

    for i in range(len(module2id) + 1):
        data['VEN_Module_Id_' + str(i)] = data['module'].apply(lambda x: x[i])

    data.drop(['log', 'module', 'module_cause'], axis=1, inplace=True)
    return data
