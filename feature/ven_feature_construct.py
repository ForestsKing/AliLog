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


def deal(string):
    string = list(filter(lambda x: x != "", string.split('module')))
    values = []
    for s in string:
        if s[0].isdigit():
            s = s.split(',')
            value = 'module:{0} cod1:{1} cod2:{2}'.format(s[0], s[1].split(':')[-1], s[2].split(':')[-1])
            values.append(value)
    if len(values) == 0:
        values.append("None")
    return values


def VenFeatureConstruct(log, submit, train=True):
    data = submit.copy(deep=True)
    log['fault_time'] = pd.to_datetime(log['fault_time'])
    data['fault_time'] = pd.to_datetime(data['fault_time'])

    data['log'] = data.apply(lambda x: getLog(log, x['sn'], x['fault_time']), axis=1)

    data['VEN_delta'] = data['log'].apply(lambda x: x[0])
    data['module'] = data['log'].apply(lambda x: x[1])
    data['module'] = data['module'].apply(lambda x: deal(x))

    if train:
        moduledf = pd.DataFrame(np.hstack(data['module'].values.tolist()), columns=['module'])
        moduledf.drop_duplicates(inplace=True)
        moduledf.sort_values('module', inplace=True)
        module2id = {}
        for i, module in enumerate(moduledf['module'].values.tolist()):
            module2id[module] = i
        joblib.dump(module2id, './user_data/model_data/ven_module2id.pkl')
    else:
        module2id = joblib.load('./user_data/model_data/ven_module2id.pkl')

    data['module'] = data['module'].apply(lambda x: list(map(lambda a: module2id.get(a, len(module2id)), x)))
    data['module'] = data['module'].apply(lambda x: (np.eye(len(module2id) + 1)[x]).max(axis=0))

    for i in range(len(module2id) + 1):
        data['VEN_Module_Id_' + str(i)] = data['module'].apply(lambda x: x[i])

    data.drop(['log', 'module'], axis=1, inplace=True)
    data.to_csv('./user_data/tmp_data/ven_feature.csv', index=False)
    return data
