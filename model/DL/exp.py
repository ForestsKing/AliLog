import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.DL.dataset import Dataset
from model.DL.earlystoping import EarlyStopping
from model.DL.model import Model
from model.DL.setseed import set_seed


class Exp:
    def __init__(self, epochs=100, batch_size=32, patience=7, lr=0.0001, random_seed=42):
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.lr = lr
        self.kf = KFold(n_splits=10, shuffle=True, random_state=random_seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(random_seed)

    def _train(self, X_train, y_train, X_valid, y_valid, d_input, model_path, k):
        dataset_train = Dataset(X_train, y_train)
        dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)
        dataset_valid = Dataset(X_valid, y_valid)
        dataloader_valid = DataLoader(dataset_valid, batch_size=self.batch_size, shuffle=True)

        model = Model(d_input=d_input).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.75 ** ((epoch - 1) // 2))
        early_stopping = EarlyStopping(patience=self.patience, verbose=True, path=model_path)
        criterion = nn.CrossEntropyLoss()

        for e in range(self.epochs):
            model.train()
            train_loss = []
            for (batch_x, batch_y) in tqdm(dataloader_train):
                optimizer.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.to(torch.int64).to(self.device)
                out = model(batch_x)
                loss = criterion(out, batch_y)
                train_loss.append(loss.item())

                loss.backward()
                optimizer.step()

            model.eval()
            valid_loss = []
            for (batch_x, batch_y) in dataloader_valid:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.to(torch.int64).to(self.device)
                out = model(batch_x)
                loss = criterion(out, batch_y)
                valid_loss.append(loss.item())

            train_loss = np.sqrt(np.average(np.array(train_loss) ** 2))
            valid_loss = np.sqrt(np.average(np.array(valid_loss) ** 2))

            print("Fold: {0} | Epoch: {1} || Train Loss: {2:.6f} | Valid Loss: {3:.6f} ".format(
                k, e, train_loss, valid_loss))

            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                break
            scheduler.step()
        model.load_state_dict(torch.load(model_path))
        return model

    def _predict(self, X_test, model):
        y_test = np.zeros(len(X_test))
        dataset_test = Dataset(X_test, y_test)
        dataloader_test = DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False, drop_last=False)

        outs = []
        for (batch_x, batch_y) in tqdm(dataloader_test):
            batch_x = batch_x.float().to(self.device)
            out = model(batch_x)
            outs.extend(out.detach().cpu().numpy().argmax(axis=1))

        return np.array(outs)

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
            model1_path = './user_data/model_data/dl/model1_' + str(k) + '.pkl'
            model2_path = './user_data/model_data/dl/model2_' + str(k) + '.pkl'
            model3_path = './user_data/model_data/dl/model3_' + str(k) + '.pkl'

            # 区分01->1, 23->0
            print('训练模型1...')
            print('---------------------------------------------------')
            y_train_copy, y_valid_copy = y_train.copy(), y_valid.copy()
            y_train_copy[(y_train_copy == 0) | (y_train_copy == 1)] = 1
            y_train_copy[(y_train_copy == 2) | (y_train_copy == 3)] = 0
            y_valid_copy[(y_valid_copy == 0) | (y_valid_copy == 1)] = 1
            y_valid_copy[(y_valid_copy == 2) | (y_valid_copy == 3)] = 0
            model1 = self._train(X_train, y_train_copy, X_valid, y_valid_copy, len(feature), model1_path, k)

            # 区分0->1, 1->0
            print('训练模型2...')
            print('---------------------------------------------------')
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
            model2 = self._train(X_train_copy, y_train_copy, X_valid_copy, y_valid_copy, len(feature), model2_path, k)

            # 区分2->1, 3->0
            print('训练模型3...')
            print('---------------------------------------------------')
            X_train_copy = X_train[(y_train == 2) | (y_train == 3)]
            X_valid_copy = X_valid[(y_valid == 2) | (y_valid == 3)]
            y_train_copy = y_train[(y_train == 2) | (y_train == 3)]
            y_valid_copy = y_valid[(y_valid == 2) | (y_valid == 3)]
            y_train_copy[y_train_copy == 2] = 1
            y_train_copy[y_train_copy == 3] = 0
            y_valid_copy[y_valid_copy == 2] = 1
            y_valid_copy[y_valid_copy == 3] = 0
            model3 = self._train(X_train_copy, y_train_copy, X_valid_copy, y_valid_copy, len(feature), model3_path, k)

            # 验证
            pred = self._predict(X_valid, model1)
            pred01 = self._predict(X_valid[pred == 1], model2)
            pred01[pred01 == 0] = 2
            pred01[pred01 == 1] = 0
            pred01[pred01 == 2] = 1
            pred23 = self._predict(X_valid[pred == 0], model3)
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
            print('第 {0} 折 valid score: {1:.4f}'.format(k, macro_F1))
            print("===================================================\n")

    def test(self, data, submit):
        feature = joblib.load('./user_data/model_data/feature.pkl')
        # 测试
        preds = []
        X_test = data[feature].fillna(0).values
        for k in range(10):
            print('第 {0} 折...'.format(k))
            print('----------------------------------------------------------')
            model1_path = './user_data/model_data/dl/model1_' + str(k) + '.pkl'
            model2_path = './user_data/model_data/dl/model2_' + str(k) + '.pkl'
            model3_path = './user_data/model_data/dl/model3_' + str(k) + '.pkl'

            model1 = Model(d_input=len(feature)).to(self.device)
            model1.load_state_dict(torch.load(model1_path, map_location=self.device))
            model2 = Model(d_input=len(feature)).to(self.device)
            model2.load_state_dict(torch.load(model2_path, map_location=self.device))
            model3 = Model(d_input=len(feature)).to(self.device)
            model3.load_state_dict(torch.load(model3_path, map_location=self.device))

            pred = self._predict(X_test, model1)
            pred01 = self._predict(X_test[pred == 1], model2)
            pred01[pred01 == 0] = 2
            pred01[pred01 == 1] = 0
            pred01[pred01 == 2] = 1
            pred23 = self._predict(X_test[pred == 0], model3)
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
