import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
sys.path.append("/Users/yohei.moriya/src/")
import rakus_ml_training as rmt


'''
演習問題：１

'''


def fit(vec_x, vec_w):
    """
    線形回帰のモデルを組み立てて値を返します。

    :param vec_x:
    :param vec_w:
    :return:
    """
    if vec_x.shape[1] != vec_w.shape[0]:
        print(f"vec_x : {vec_x.shape}, vec_w : {vec_w.shape}")
        raise Exception('引数が不正です。')

    vec_y = vec_x.dot(vec_w.T)
    return vec_y


def cal_error(vec_y, vec_t):
    """
    出力の誤差を計算します

    :param vec_y:
    :param vec_t:
    :return:
    """
    if vec_y.shape[0] != vec_t.shape[0]:
        print(f"vec_y : {vec_y.shape} , vec_t : {vec_t.shape}")
        raise Exception('引数が不正です。')
    vec_dev = vec_y - vec_t
    error = np.sqrt((np.mean(vec_dev ** 2)) / 2)
    #     error = (np.mean(vec_dev ** 2)) / 2
    return -error


def cal_derror(vec_x, vec_w, vec_t):
    """
    誤差の勾配を計算します。

    :param vec_x:
    :param vec_w:
    :param vec_t:
    :return:
    """
    if vec_x.shape[1] != vec_w.shape[0] and vec_x.shape[0] != vec_t[0]:
        print(f"vec_x : {vec_x.shape} , vec_w : {vec_w.shape} , vec_t : {vec_t.shape}")
        raise Exception('引数が不正です。')

    vec_y = fit(vec_x, vec_w)
    vec_dev = vec_y - vec_t
    vec_dcost = []
    for index_x in range(vec_x.shape[1]):
        vec_dcost.append(vec_dev * vec_x[:, index_x])

    return 2 * np.mean(np.array(vec_dcost), axis=1)


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
    df_choose = df_origin.drop('CHAS', axis=1)
    return df_choose


def do_cleansing(df_train, df_test):
    """
    入力されたデータを結合し、まとめて前処理を実行します。
    前処理が終了すると、元のindex番号を基準にして、データフレームを分割することで、データ集合を復元します。

    :param df_train:
    :param df_test:
    :return:
    """
    if df_train.shape[1] != df_test.shape[1]:
        raise Exception('入力されたデータの特徴量数が一致しません')

    df_all = pd.concat([df_train, df_test])

    df_all_clean = df_all.copy()

    """
    前処理を列挙する
    """
    # -----------------------------------------------------------------------------------------
    #     特徴量の選択
    #     df_all_clean = choose(df_all_clean)

    #     特徴量の２乗を計算
    df_all_clean = pd.concat([df_all_clean, square(df_all_clean)], axis=1)
    #     print(df_all_clean.columns)

    #     特徴量の２乗と3乗を計算
    #     df_all_clean = pd.concat([df_all_clean, square(df_all_clean), cube(df_all_clean)], axis=1)
    #     print(df_all_clean.columns)

    #     標準化
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


def run(df_train, df_label, count):
    """
    回帰を実行します。

    :param df_train:
    :param df_label:
    :param count
    :return:
    """
    nd_train = df_train.values
    nd_label = df_label.values

    vec_w = np.zeros(nd_train.shape[1])

    #     print(nd_train.shape)
    #     print(vec_w.shape)

    for i in range(count):
        vec_w = train(nd_train, vec_w, nd_label, 0.01)
        y = fit(nd_train, vec_w)
        error = cal_error(y, df_label.values)
        print(f"{i} : {error}")
        # TODO 勾配の差を閾値にする
        if np.abs(error) < 2:
            break

    return vec_w


if __name__ == '__main__':

    df_test_origin = rmt.boston.get_test_data()
    df_train_origin = rmt.boston.get_train_data()

    df_train = df_train_origin.drop('TARGET', axis=1)
    df_test = df_test_origin
    df_label = df_train_origin['TARGET']

    df_train_clean, df_test_clean = do_cleansing(df_train.copy(), df_test.copy())

    df_train_clean['BIAS'] = 1
    df_test_clean['BIAS'] = 1

    vec_w = run(df_train_clean, df_label, 100000)

    print(f"vec_w")
    print(f"{vec_w}")

    nd_result = fit(vec_w=vec_w, vec_x=df_test_clean.values)
    df_result = pd.DataFrame(data=nd_result, columns=['TARGET'])
    # rmt.boston.confirm(df_result)

