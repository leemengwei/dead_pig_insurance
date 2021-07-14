import copy
import sklearn
from sklearn import ensemble
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


def GradientBoosting(df, df1, var, st, ed, label):
    # X_train_ori, X_test_ori, y_train_ori, y_test_ori = \
    #     train_test_split(np.array(df.iloc[:, st:ed]), np.array(df.iloc[:, label]), test_size=0.3)
    X_train_ori = np.array(df.iloc[:, st:ed])
    y_train_ori = np.array(df.iloc[:, label])
    ss = MinMaxScaler()
    X_train = ss.fit_transform(X_train_ori.reshape(-1, 1)).reshape(-1, ed - st)
    # X_test = ss.fit_transform(X_test_ori.reshape(-1, 1)).reshape(-1, ed - st)
    y_train = ss.fit_transform(y_train_ori.reshape(-1, 1))
    # y_test = ss.fit_transform(y_test_ori.reshape(-1, 1))

    # 训练程序
    # if var == 'Weight':
    #     GBRT = ensemble.GradientBoostingRegressor(random_state=10)
    #     pipe = Pipeline([("scaler", MinMaxScaler()), ("GBRT", GBRT)])
    #     tuned_parameters = {'GBRT__learning_rate': np.linspace(0.1, 1, 10),
    #                         'GBRT__max_depth': range(1, 11, 1), 'GBRT__max_leaf_nodes': range(2, 11, 1)}
    #     grid = GridSearchCV(pipe, tuned_parameters, cv=5)
    #     grid = grid.fit(X_train_ori, y_train_ori.reshape(-1, 1).ravel())
    #     print(grid.best_params_)
    #     model_GradientBoostingRegressor_1 = ensemble.GradientBoostingRegressor(
    #         max_depth=grid.best_params_['GBRT__max_depth'], max_leaf_nodes=grid.best_params_['GBRT__max_leaf_nodes'],
    #         learning_rate=grid.best_params_['GBRT__learning_rate'], random_state=10).fit(X_train, y_train.ravel())
    model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(max_depth=1, random_state=10).fit(X_train, y_train.ravel())

    # print("ok:", np.array(df1.iloc[:, st:ed]))
    if var == 'Weight':
        predict_GradientBoostingRegressor = model_GradientBoostingRegressor.predict(X_train)
    else:
        predict_GradientBoostingRegressor = model_GradientBoostingRegressor.predict(X_train)
    y_predict = ss.inverse_transform(predict_GradientBoostingRegressor.reshape(-1, 1)).squeeze()
    lst = df.index.tolist()
    lst1 = df1.index.tolist()
    print(lst)
    if var == 'Length':
        df_ori = pd.read_csv('dead_pig.csv')
        cnt = 0
        df_ori['predict_length'] = np.nan
        for i in lst:
            df_ori['predict_length'].iloc[i] = y_predict[cnt]
            cnt += 1
        df_ori.to_csv('dead_pig_predict.csv')
    if var == 'Weight':
        df_ori = pd.read_csv('dead_pig_predict.csv')
        cnt = 0
        df_ori['predict_weight'] = np.nan
        for i in lst:
            df_ori['predict_weight'].iloc[i] = y_predict[cnt]
            cnt += 1
        df_ori = df_ori.loc[:, ~df_ori.columns.str.contains('^Unnamed')]
        df_ori.to_csv('dead_pig_predict.csv')
    y_test = ss.inverse_transform(y_train.reshape(-1, 1)).squeeze()
    # 相对精度
    # print(y_predict)
    # relative_accuracy = (1 - (np.abs(y_test - y_predict) / y_test)).mean()
    # print("relative accuracy:", relative_accuracy)
    # 决定系数
    # GradientBoostingRegressor_score = r2_score(y_test, y_predict)
    # print(var + "GBRT R^2:", GradientBoostingRegressor_score)

    # predict_GradientBoostingRegressor = model_GradientBoostingRegressor.predict(X_test)
    # 均方误差
    # GradientBoostingRegressor_score = mean_squared_error(y_test, predict_GradientBoostingRegressor)
    # print(GradientBoostingRegressor_score)
    # 绝对平均误差
    # GradientBoostingRegressor_score = mean_absolute_error(y_test, predict_GradientBoostingRegressor)
    # print(GradientBoostingRegressor_score)
    # y_predict = ss.inverse_transform(predict_GradientBoostingRegressor.reshape(-1, 1)).squeeze()
    # y_test = ss.inverse_transform(y_test.reshape(-1, 1)).squeeze()
    # 相对精度
    # relative_accuracy = 1 - (np.abs(y_test - y_predict) / y_test)
    # 决定系数
    # GradientBoostingRegressor_score = r2_score(y_test, y_predict)
    # print(var + "GBRT R^2:", GradientBoostingRegressor_score)
    # plt.clf()  # 清理历史绘图
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # x = range(1, len(y_test) + 1)
    # plt.bar(x, height=y_test, width=0.5, align='center', color='#244FFE', edgecolor='#244FFE')
    # plt.plot(x, y_predict, "-o", color='#CD3834')
    # plt.grid()

    plt.clf()  # 清理历史绘图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    y_train = ss.inverse_transform(y_train.reshape(-1, 1)).squeeze()
    x = range(1, len(y_train) + 1)
    fig1, axes1 = plt.subplots()
    axes1.bar(x, height=y_train, width=0.5, align='center', color='#244FFE', edgecolor='#244FFE')
    axes1.plot(x, y_predict, "-o", color='#CD3834')
    axes1.grid()

    fig2, axes2 = plt.subplots()
    df_box = pd.DataFrame(y_predict - y_train)
    df_box.plot.box(ax=axes2)
    fig2.savefig('GradientBoosting' + var + 'Box.png', dpi=100, bbox_inches='tight')
    # df_box = pd.DataFrame(y_predict - y_train)
    # plt.boxplot(df_box)
    # plt.savefig('GradientBoosting' + var + 'Box.png', dpi=100, bbox_inches='tight')

    # plt.show()
    fig1.savefig('GradientBoosting' + var + '.png', dpi=100, bbox_inches='tight')


def LR(df, var, st, ed, label):
    X_train_ori, X_test_ori, y_train_ori, y_test_ori = \
        train_test_split(np.array(df.iloc[:, st:ed]), np.array(df.iloc[:, label]), test_size=0.3)
    ss = MinMaxScaler()
    X_train = ss.fit_transform(X_train_ori.reshape(-1, 1)).reshape(-1, ed - st)
    X_test = ss.fit_transform(X_test_ori.reshape(-1, 1)).reshape(-1, ed - st)
    y_train = ss.fit_transform(y_train_ori.reshape(-1, 1))
    y_test = ss.fit_transform(y_test_ori.reshape(-1, 1))

    model_lr = LinearRegression().fit(X_train, y_train)

    # predict_LRRegressor = model_lr.predict(X_test)

    predict_LRRegressor = model_lr.predict(X_train)
    y_predict = ss.inverse_transform(predict_LRRegressor.reshape(-1, 1)).squeeze()
    y_test = ss.inverse_transform(y_train.reshape(-1, 1)).squeeze()
    relative_accuracy = (1 - (np.abs(y_test - y_predict) / y_test)).mean()
    print("relative accuracy:", relative_accuracy)
    # 决定系数
    LRRegressor_score = r2_score(y_test, y_predict)
    print(var + "LR R^2:", LRRegressor_score)

    # 均方误差
    # LRRegressor_score = mean_squared_error(y_test, predict_LRRegressor)
    # print(LRRegressor_score)
    # 绝对平均误差
    # LRRegressor_score = mean_absolute_error(y_test, predict_LRRegressor)
    # print(LRRegressor_score)

    # y_predict = ss.inverse_transform(predict_LRRegressor.reshape(-1, 1)).squeeze()
    # y_test = ss.inverse_transform(y_test.reshape(-1, 1)).squeeze()
    # temp = (1 - (np.abs(y_test - y_predict) / y_test)).mean()
    # print("OK:", temp)
    # # 决定系数
    # LRRegressor_score = r2_score(y_test, y_predict)
    # print(var + "LR:", LRRegressor_score)

    # plt.clf()  # 清理历史绘图
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # x = range(1, len(y_test) + 1)
    # plt.bar(x, height=y_test, width=0.5, align='center', color='#244FFE', edgecolor='#244FFE')
    # plt.plot(x, y_predict, "-o", color='#CD3834')
    # plt.grid()

    plt.clf()  # 清理历史绘图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    y_train = ss.inverse_transform(y_train.reshape(-1, 1)).squeeze()
    x = range(1, len(y_train) + 1)
    plt.bar(x, height=y_train, width=0.5, align='center', color='#244FFE', edgecolor='#244FFE')
    plt.plot(x, y_predict, "-o", color='#CD3834')
    plt.grid()

    # plt.show()
    plt.savefig('LR' + var + '.png', dpi=100, bbox_inches='tight')


if __name__ == '__main__':
    df = pd.read_csv('dead_pig.csv')
    lst = ['file_name_top', 'file_name_side',
           'nose_coordinate_top', 'ear_coordinate_top', 'hip_coordinate_top', 'tail_coordinate_top',
           'left_top_coordinate_top', 'left_bottom_coordinate_top', 'right_top_coordinate_top', 'right_bottom_coordinate_top',
           'nose_coordinate_side', 'ear_coordinate_side', 'hip_coordinate_side', 'tail_coordinate_side',
           'left_top_coordinate_side', 'left_bottom_coordinate_side', 'right_top_coordinate_side', 'right_bottom_coordinate_side']
    df = df.drop(lst, axis=1)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.dropna(axis=0, how='any')
    # df.fillna(method='ffill', inplace=True)  # 前值填充
    # df.fillna(method='bfill', inplace=True)  # 后值填充
    df_length = copy.deepcopy(df)
    # print(df_length)
    for index, row in df.iterrows():
        if df['weight'][index] == 0:
            df['weight'][index] = np.nan
    df = df.dropna(axis=0, how='any')
    df_weight = copy.deepcopy(df)
    df_weight = df_weight.drop('length', axis=1)
    df_length = df_length.drop('weight', axis=1)
    # 三个数分别是 训练数据起始列index 训练数据结束列index+1 标签列index
    GradientBoosting(df_length, df_length, 'Length', 0, 8, 8)
    GradientBoosting(df_weight, df_length, 'Weight', 0, 8, 8)
    # LR(df_length, 'Length', 0, 8, 8)
    # LR(df_weight, 'Weight', 0, 8, 8)
