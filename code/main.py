import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))

import pandas as pd

from feature.sel_feature_construct import SelFeatureConstruct
from feature.ven_feature_construct import VenFeatureConstruct
from feature.cra_feature_construct import CraFeatureConstruct
from model.catboost import CatBoost

TRAIN = False
TEST = True
if __name__ == '__main__':
    # 训练
    if TRAIN:
        print("正在训练...")
        train_submit = pd.concat([pd.read_csv('./data/preliminary_train_label_dataset.csv'),
                                  pd.read_csv('./data/preliminary_train_label_dataset_s.csv')], ignore_index=True)
        print(' 共需处理 {0} 个异常'.format(len(train_submit)))

        print(" SEL 特征构造...")
        train_sel_log = pd.read_csv('./data/preliminary_sel_log_dataset.csv')
        print(' 共需处理 {0} 条 SEL 日志'.format(len(train_sel_log)))
        train_sel = SelFeatureConstruct(train_sel_log, train_submit, train=True)

        print(" VEN 特征构造...")
        train_ven_log = pd.read_csv('./data/preliminary_venus_dataset.csv')
        print(' 共需处理 {0} 条 VEN 日志'.format(len(train_ven_log)))
        train_ven = VenFeatureConstruct(train_ven_log, train_submit, train=True)

        print(" CRA 特征构造...")
        train_cra_log = pd.read_csv('./data/preliminary_crashdump_dataset.csv')
        print(' 共需处理 {0} 条 CRA 日志'.format(len(train_cra_log)))
        train_cra = CraFeatureConstruct(train_cra_log, train_submit, train=True)

        print(" 特征拼接...")
        train_ven.drop(['sn', 'fault_time', 'label'], axis=1, inplace=True)
        train_cra.drop(['sn', 'fault_time', 'label'], axis=1, inplace=True)
        train = pd.concat([train_sel, train_ven, train_cra], axis=1)

        print(" 模型训练...")
        model = CatBoost()
        model.train(train)

    # 测试
    if TEST:
        print("正在测试...")
        test_submit = pd.read_csv('./tcdata/final_submit_dataset_a.csv')
        print(' 共需处理 {0} 个异常'.format(len(test_submit)))

        print(" SEL 特征构造...")
        test_sel_log = pd.read_csv('./tcdata/final_sel_log_dataset_a.csv')
        print(' 共需处理 {0} 条 SEL 日志'.format(len(test_sel_log)))
        test_sel = SelFeatureConstruct(test_sel_log, test_submit, train=False)

        print(" VEN 特征构造...")
        test_ven_log = pd.read_csv('./tcdata/final_venus_dataset_a.csv')
        print(' 共需处理 {0} 条 VEN 日志'.format(len(test_ven_log)))
        test_ven = VenFeatureConstruct(test_ven_log, test_submit, train=False)

        print(" CRA 特征构造...")
        test_cra_log = pd.read_csv('./tcdata/final_crashdump_dataset_a.csv')
        print(' 共需处理 {0} 条 CRA 日志'.format(len(test_cra_log)))
        test_cra = CraFeatureConstruct(test_cra_log, test_submit, train=False)

        print(" 特征拼接...")
        test_ven.drop(['sn', 'fault_time'], axis=1, inplace=True)
        test_cra.drop(['sn', 'fault_time'], axis=1, inplace=True)
        test = pd.concat([test_sel, test_ven, test_cra], axis=1)

        print(" 模型测试...")
        model = CatBoost()
        result = model.test(test, test_submit)
        result[['sn', 'fault_time', 'label']].to_csv('./prediction_result/predictions.csv', index=False)
        print("完成测试!")
