import joblib
import pandas as pd


def getLog(log, sn, fault_time):
    tmp = log[(log['sn'] == sn)].copy(deep=True)
    tmp['delta'] = tmp['fault_time'].apply(lambda x: abs((fault_time - x)))
    if len(tmp) > 0:
        tmp = tmp[tmp['delta'] < pd.to_timedelta("1D")]
        if len(tmp) > 0:
            tmp['delta'] = tmp['fault_time'].apply(lambda x: (fault_time - x).total_seconds())
            tmp = tmp.sort_values('delta')
            return [tmp['delta'].values[0], tmp['fault_code'].values[0]]
        else:
            return [0, ""]
    else:
        return [0, ""]


def CraFeatureConstruct(log, submit, train=True):
    data = submit.copy(deep=True)
    log['fault_time'] = pd.to_datetime(log['fault_time'])
    data['fault_time'] = pd.to_datetime(data['fault_time'])

    data['log'] = data.apply(lambda x: getLog(log, x['sn'], x['fault_time']), axis=1)
    data['CRA_delta'] = data['log'].apply(lambda x: x[0])
    data['fault_code'] = data['log'].apply(lambda x: x[1])

    data['CRA_fault_code_0'] = data['fault_code'].apply(lambda x: x.split('.')[0] if x != '' else '')
    data['CRA_fault_code_1'] = data['fault_code'].apply(lambda x: x.split('.')[1] if x != '' else '')
    data['CRA_fault_code_2'] = data['fault_code'].apply(lambda x: x.split('.')[2] if x != '' else '')

    if train:
        fc02id = {}
        for i, fc in enumerate(sorted(list(set(data['CRA_fault_code_0'].values.tolist())))):
            fc02id[fc] = i
        joblib.dump(fc02id, './user_data/model_data/cra_fc02id.pkl')

        fc12id = {}
        for i, fc in enumerate(sorted(list(set(data['CRA_fault_code_1'].values.tolist())))):
            fc12id[fc] = i
        joblib.dump(fc12id, './user_data/model_data/cra_fc12id.pkl')

        fc22id = {}
        for i, fc in enumerate(sorted(list(set(data['CRA_fault_code_2'].values.tolist())))):
            fc22id[fc] = i
        joblib.dump(fc22id, './user_data/model_data/cra_fc22id.pkl')
    else:
        fc02id = joblib.load('./user_data/model_data/cra_fc02id.pkl')
        fc12id = joblib.load('./user_data/model_data/cra_fc12id.pkl')
        fc22id = joblib.load('./user_data/model_data/cra_fc22id.pkl')

    data['CRA_fault_code_0'] = data['CRA_fault_code_0'].apply(lambda x: fc02id.get(x, len(fc02id)))
    data['CRA_fault_code_1'] = data['CRA_fault_code_1'].apply(lambda x: fc12id.get(x, len(fc12id)))
    data['CRA_fault_code_2'] = data['CRA_fault_code_2'].apply(lambda x: fc22id.get(x, len(fc22id)))

    data.drop(['log', 'fault_code'], axis=1, inplace=True)
    data.to_csv('./user_data/tmp_data/cra_feature.csv', index=False)
    return data
