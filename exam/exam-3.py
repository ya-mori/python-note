import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
sys.path.append("/Users/yohei.moriya/src/")
import rakus_ml_training as rmt


"""
演習問題3

"""


def softmax(nd_x: pd.DataFrame)->np.ndarray:
    """
    ソフトマックス関数に入力した値を計算します。

    :param nd_x:
    :return:
    """
    ans = np.exp(nd_x) / np.sum(np.exp(nd_x), axis=1).reshape((nd_x.shape[0], 1))
    return ans


def fit(vec_x, vec_w)->np.ndarray:
    """
    モデルを組み立てて値を返します。

    :param vec_x:
    :param vec_w:
    :return:
    """
    if vec_x.shape[1] != vec_w.shape[1]:
        print(f"vec_x : {vec_x.shape}, vec_w : {vec_w.shape}")
        raise Exception('引数が不正です。')

    vec_z = vec_x.dot(vec_w.T)
    return softmax(vec_z)


def cal_error(vec_y, vec_t):
    """
    出力の誤差を計算します

    :param vec_y:
    :param vec_t:
    :return:
    """
    if vec_y.shape[0] != vec_t.shape[0] or vec_y.shape[0] != vec_t.shape[0]:
        print(f"vec_y : {vec_y.shape} , vec_t : {vec_t.shape}")
        raise Exception('引数が不正です。')

    vec_item = vec_t * np.log(vec_y)
    error = np.sum(vec_item, axis=1)
    return np.mean(error)


def cal_derror(vec_x, vec_w, vec_t):
    """
    誤差の勾配を計算します。

    :param vec_x:
    :param vec_w:
    :param vec_t:
    :return:
    """
    if vec_x.shape[1] != vec_w.shape[1] or vec_w.shape[0] != vec_t.shape[1]:
        print(f"vec_x : {vec_x.shape} , vec_w : {vec_w.shape} , vec_t : {vec_t.shape}")
        raise Exception('引数が不正です。')

    vec_y = fit(vec_x, vec_w)
    nd_dev = vec_y - vec_t
    # vec_dcost = nd_dev.T.dot(vec_x)
    vec_dcost = nd_dev.T.dot(vec_x)

    nd_dcost = vec_dcost
    return nd_dcost


def train(vec_x, vec_w, vec_t, alpha):
    """
    ハイパーパラメータを更新します。

    :param vec_x:
    :param vec_w:
    :param vec_t:
    :param alpha:
    :return:
    """
    vec_item = cal_derror(vec_x, vec_w, vec_t) * alpha
    vec_re_w = vec_w - vec_item
    return vec_re_w


def norm(df_origin):
    """
    行列を正規化します。
    """
    vec_mean = df_origin.mean()
    vec_var = df_origin.var()
    nd_train = df_origin.values
    nd_norm = (nd_train - vec_mean.values) / np.sqrt(vec_var).values
    return pd.DataFrame(data=nd_norm, columns=df_origin.columns, index=df_origin.index)


def square(df_origin):
    """
    行列の二乗を計算します
    """
    df_origin_sq = df_origin ** 2
    df_origin_sq = df_origin_sq.rename(columns=lambda name: name + "^2")
    return df_origin_sq


def cube(df_origin):
    """
    行列の三乗を計算します
    """
    df_origin_cube = df_origin ** 3
    df_origin_cube = df_origin_cube.rename(columns=lambda name: name + "^3")
    return df_origin_cube


def choose(df_origin):
    """
    特徴量を選択します

    :param df_origin:
    :return:
    """
    df_choose = df_origin.drop('sepal width (cm)', axis=1)
    return df_choose


def do_cleansing(df_train, df_test):

    if df_train.shape[1] != df_test.shape[1]:
        raise Exception('入力されたデータの特徴量数が一致しません')

    df_all = pd.concat([df_train, df_test])

    df_all_clean = df_all.copy()

    """
    前処理を列挙する
    """
    # -----------------------------------------------------------------------------------------
    # 特徴量の選択
    # df_all_clean = choose(df_all_clean)

    # 特徴量の２乗を計算
    # df_all_clean = pd.concat([df_all_clean, square(df_all_clean)], axis=1)
    # print(df_all_clean.columns)

    # 特徴量の２乗と3乗を計算
    # df_all_clean = pd.concat([df_all_clean, square(df_all_clean), cube(df_all_clean)], axis=1)
    # print(df_all_clean.columns)

    # 標準化
    df_all_clean = norm(df_all_clean)
    # -----------------------------------------------------------------------------------------

    #  分割
    df_train_clean = df_all_clean.iloc[:df_train.shape[0], :]
    df_test_clean = df_all_clean.iloc[df_train.shape[0]:, :]

    if df_train_clean.shape[0] == df_train.shape[0]:
        return df_train_clean, df_test_clean
    else:
        print(f"{df_train_clean.shape[0]}, {df_train.shape[0]}")
        raise Exception('前処理した結果データの数に不整合が生じたようです。')


def run(df_train: pd.DataFrame, df_label: pd.DataFrame, count: int):
    """
    分類を実行します。

    :param df_train:
    :param df_label:
    :return:
    """
    if df_train.shape[0] != df_label.shape[0]:
        raise Exception('学習データとラベルデータの長さが一致していません。')

    nd_w = np.zeros((df_label.shape[1], df_train.shape[1]))

    for i in range(count):
        nd_w = train(df_train.values, nd_w, df_label.values, 0.01)
        h = fit(df_train.values, nd_w)
        error = cal_error(h, df_label.values)
        print(f"{i} : error {error}")

    return nd_w


if __name__ == '__main__':

    df_train_origin = rmt.iris.get_train_data()
    df_test_origin = rmt.iris.get_test_data()

    df_train = df_train_origin.drop('target', axis=1)
    df_test = df_test_origin
    df_label = pd.get_dummies(df_train_origin['target'])

    df_train_clean, df_test_clean = do_cleansing(df_train, df_test)

    df_train_clean['bias'] = 1
    df_test_clean['bias'] = 1

    nd_w = run(df_train_clean, df_label, 10000)

    nd_result = fit(df_train_clean.values, nd_w)
    df_result = pd.DataFrame(data=nd_result)
    print('train result')
    rmt.iris.confirm(df_result.idxmax(axis=1), df_label.idxmax(axis=1))

    nd_result = fit(df_test_clean.values, nd_w)
    df_result = pd.DataFrame(data=nd_result)
    print('test result')
    rmt.iris.confirm(df_result.idxmax(axis=1))
