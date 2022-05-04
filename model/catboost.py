import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold


class CatBoost:
    def __init__(self):
        self.random_seed = 42
        self.kf = KFold(n_splits=10, shuffle=True, random_state=self.random_seed)

    def train(self, data):
        feature = data.columns.values.tolist()
        feature.remove('sn')
        feature.remove('fault_time')
        feature.remove('label')
        joblib.dump(feature, './user_data/model_data/feature.pkl')

        X = data[feature].fillna(0).values
        y = data['label'].values

        for k, (train_index, valid_index) in enumerate(self.kf.split(X, y)):
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]

            # 区分01->1, 23->0
            y_train_copy, y_valid_copy = y_train.copy(), y_valid.copy()
            y_train_copy[(y_train_copy == 0) | (y_train_copy == 1)] = 1
            y_train_copy[(y_train_copy == 2) | (y_train_copy == 3)] = 0
            y_valid_copy[(y_valid_copy == 0) | (y_valid_copy == 1)] = 1
            y_valid_copy[(y_valid_copy == 2) | (y_valid_copy == 3)] = 0

            model1 = CatBoostClassifier(loss_function='Logloss', verbose=False, eval_metric='F1',
                                        class_weights=[1 / 2, 1 / 2], random_seed=self.random_seed,
                                        learning_rate=0.1, use_best_model=True, train_dir='./user_data/model_data/log/')
            model1.fit(X_train, y_train_copy, eval_set=(X_valid, y_valid_copy), plot=False)

            # 区分0->1, 1->0
            X_train_copy = X_train[(y_train == 0) | (y_train == 1)]
            X_valid_copy = X_valid[(y_valid == 0) | (y_valid == 1)]
            y_train_copy = y_train[(y_train == 0) | (y_train == 1)]
            y_valid_copy = y_valid[(y_valid == 0) | (y_valid == 1)]
            y_train_copy[y_train_copy == 0] = 2
            y_train_copy[y_train_copy == 1] = 0
            y_train_copy[y_train_copy == 2] = 1
            y_valid_copy[y_valid_copy == 0] = 2
            y_valid_copy[y_valid_copy == 1] = 0
            y_valid_copy[y_valid_copy == 2] = 1

            model2 = CatBoostClassifier(loss_function='Logloss', verbose=False, eval_metric='F1',
                                        class_weights=[1 / 2, 1 / 2], random_seed=self.random_seed,
                                        learning_rate=0.1, use_best_model=True, train_dir='./user_data/model_data/log/')
            model2.fit(X_train_copy, y_train_copy, eval_set=(X_valid_copy, y_valid_copy), plot=False)

            # 区分2->1, 3->0
            X_train_copy = X_train[(y_train == 2) | (y_train == 3)]
            X_valid_copy = X_valid[(y_valid == 2) | (y_valid == 3)]
            y_train_copy = y_train[(y_train == 2) | (y_train == 3)]
            y_valid_copy = y_valid[(y_valid == 2) | (y_valid == 3)]
            y_train_copy[y_train_copy == 2] = 1
            y_train_copy[y_train_copy == 3] = 0
            y_valid_copy[y_valid_copy == 2] = 1
            y_valid_copy[y_valid_copy == 3] = 0

            model3 = CatBoostClassifier(loss_function='Logloss', verbose=False, eval_metric='F1',
                                        class_weights=[1 / 2, 1 / 2], random_seed=self.random_seed,
                                        learning_rate=0.1, use_best_model=True, train_dir='./user_data/model_data/log/')
            model3.fit(X_train_copy, y_train_copy, eval_set=(X_valid_copy, y_valid_copy), plot=False)

            # 验证
            pred = model1.predict(X_valid)
            pred01 = model2.predict(X_valid[pred == 1])
            pred01[pred01 == 0] = 2
            pred01[pred01 == 1] = 0
            pred01[pred01 == 2] = 1
            pred23 = model3.predict(X_valid[pred == 0])
            pred23[pred23 == 0] = 3
            pred23[pred23 == 1] = 2
            pred[pred == 0] = pred23
            pred[pred == 1] = pred01

            overall_df = pd.DataFrame()
            overall_df['label_gt'] = y_valid
            overall_df['label_pr'] = pred

            weights = [5 / 11, 4 / 11, 1 / 11, 1 / 11]

            macro_F1 = 0
            for i in range(len(weights)):
                TP = len(overall_df[(overall_df['label_gt'] == i) & (overall_df['label_pr'] == i)])
                FP = len(overall_df[(overall_df['label_gt'] != i) & (overall_df['label_pr'] == i)])
                FN = len(overall_df[(overall_df['label_gt'] == i) & (overall_df['label_pr'] != i)])
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                F1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                macro_F1 += weights[i] * F1
            print('  第 {0} 折 valid score: {1:.4f}'.format(k, macro_F1))

            joblib.dump(model1, './user_data/model_data/catboost/model1_' + str(k) + '.pkl')
            joblib.dump(model2, './user_data/model_data/catboost/model2_' + str(k) + '.pkl')
            joblib.dump(model3, './user_data/model_data/catboost/model3_' + str(k) + '.pkl')

    def test(self, data, submit):
        feature = joblib.load('./user_data/model_data/feature.pkl')
        # 测试
        preds = []
        X_test = data[feature].fillna(0).values
        for i in range(10):
            model1 = joblib.load('./user_data/model_data/catboost/model1_' + str(i) + '.pkl')
            model2 = joblib.load('./user_data/model_data/catboost/model2_' + str(i) + '.pkl')
            model3 = joblib.load('./user_data/model_data/catboost/model3_' + str(i) + '.pkl')

            pred = model1.predict(X_test)
            pred01 = model2.predict(X_test[pred == 1])
            pred01[pred01 == 0] = 2
            pred01[pred01 == 1] = 0
            pred01[pred01 == 2] = 1
            pred23 = model3.predict(X_test[pred == 0])
            pred23[pred23 == 0] = 3
            pred23[pred23 == 1] = 2
            pred[pred == 0] = pred23
            pred[pred == 1] = pred01

            preds.append(pred)

        preds = np.array(preds).T
        preds1 = []
        for pred in preds:
            preds1.append(np.argmax(np.bincount(pred)))

        submit['label'] = np.array(preds1)
        return submit
