import os
import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from utils import unbiased_rmse, _bias, r2_score, _mse
from config import get_args
from utils import GetKGE, GetNSE, GetPCC, _plotloss, _plotbox, _boxkge, _boxnse, _boxpcc, GetMAE, GetRMSE, _boxbias
from loss import NaNMSELoss


def lon_transform(x):
    x_new = np.zeros(x.shape)
    x_new[:, :, :int(x.shape[2] / 2)] = x[:, :, int(x.shape[2] / 2):]
    x_new[:, :, int(x.shape[2] / 2):] = x[:, :, :int(x.shape[2] / 2)]
    return x_new


def two_dim_lon_transform(x):
    x_new = np.zeros(x.shape)
    x_new[:, :int(x.shape[1] / 2)] = x[:, int(x.shape[1] / 2):]
    x_new[:, int(x.shape[1] / 2):] = x[:, :int(x.shape[1] / 2)]
    return x_new


def postprocess(cfg):
    PATH = cfg['inputs_path'] + cfg['product'] + '/' + str(cfg['spatial_resolution']) + '/'
    file_name_mask = 'Mask with {sr} spatial resolution.npy'.format(sr=cfg['spatial_resolution'])
    mask = np.load(PATH + file_name_mask)
    # ------------------------------------------------------------------------------------------------------------------------------
    if cfg['modelname'] in ['MHLSTMModel', 'MSLSTMModel', 'SoftMTLv1', 'MMOE', 'LSTM', 'MTLCNN', 'MTLConvLSTMModel',
                            'MTANmodel', 'CrossStitchNet', 'CNN', 'ConvLSTM', 'FAMLSTM']:
        # if cfg['modelname'] == 'KDE_LSTM':
        print('KDE_MTLModel ---> ')
        out_path_mhl = cfg['inputs_path'] + cfg['product'] + '/' + str(cfg['spatial_resolution']) + '/' + cfg[
            'workname'] + '/' + cfg['modelname'] + '/focast_time ' + str(cfg['forcast_time']) + '/'
        # y_pred_mhls = np.load(out_path_mhl + '_predictions_s.npy')
        # y_test_mhls = np.load(out_path_mhl + 'observations_s.npy')
        if cfg['modelname'] in ['LSTM', 'CNN', 'ConvLSTM']:
            y_pred_mhls = np.load(out_path_mhl + '_predictions.npy')
            y_test_mhls = np.load(out_path_mhl + 'observations.npy')
        elif cfg['label'][0] in ['volumetric_soil_water_layer_1'] and cfg['label'][1] in ['total_evaporation']:
            with open(out_path_mhl + '_predictions_s.npy', 'rb') as f:
                y_pred_mhls = pickle.load(f)
            with open(out_path_mhl + 'observations_s.npy', 'rb') as f:
                y_test_mhls = pickle.load(f)

        else:
            y_pred_mhls = np.load(out_path_mhl + '_predictions_s.npy', allow_pickle=True)
            y_test_mhls = np.load(out_path_mhl + 'observations_s.npy', allow_pickle=True)

        model = torch.load((out_path_mhl + cfg['modelname'] + '_para.pkl'), map_location=torch.device('cuda:0'))

        y_pred_mhls = np.array(y_pred_mhls)
        y_test_mhls = np.array(y_test_mhls)

        # if cfg['modelname'] in ['CrossStitchNet']:
        #     for i in range(2):
        #         mask = y_test_mhls[i] == y_test_mhls[i]
        #         y_pred_mhls[i] = y_pred_mhls[i][mask]
        #         y_test_mhls[i] = y_test_mhls[i][mask]
        if cfg['modelname'] in ['LSTM', 'CNN', 'ConvLSTM']:
            nt, nlat, nlon = y_test_mhls.shape
        else:
            nt, nlat, nlon = y_test_mhls[0].shape
            # y_test_mhls[1] = np.array(y_test_mhls)[1]
            # y_test_mhls[0] = np.array(y_test_mhls)[0]

        # 将数组或者矩阵存储为csv文件可以使用如下代码实现：
        #
        # numpy.savetxt('new.csv', my_matrix, delimiter=',')

        # y_pred_mhle = np.load(out_path_mhl + '_predictions_e.npy')
        # y_test_mhle = np.load(out_path_mhl + 'observations_e.npy')

        # print('y_pred_mhls='+y_pred_mhls.shape,'y_pred_mhlv'+y_pred_mhlv.shape,'y_est'+ y_test_mhl.shape)
        # get shape

        # mask
        # mask=y_test_mhl==y_test_mhl
        # cal perf
        # r2_mhls = np.full((nlat, nlon), np.nan)
        # urmse_mhls = np.full((nlat, nlon), np.nan)
        print(model)

        r = []
        kge = []
        nse = []
        pcc = []
        loss_test = []
        r2 = []
        rmse = []
        bias = []
        lossmse = torch.nn.MSELoss()

        if cfg['modelname'] in ['LSTM', 'CNN', 'ConvLSTM']:
            r_mhls = np.full((nlat, nlon), np.nan)
            loss_time = np.full((nlat, nlon), np.nan)
            kge_time = np.full((nlat, nlon), np.nan)
            pcc_time = np.full((nlat, nlon), np.nan)
            nse_time = np.full((nlat, nlon), np.nan)
            bias_time = np.full((nlat, nlon), np.nan)
            r2_time = np.full((nlat, nlon), np.nan)
            rmse_time = np.full((nlat, nlon), np.nan)

            for i in range(nlat):
                for j in range(nlon):
                    if not (np.isnan(y_test_mhls[:, i, j]).any()):
                        r_mhls[i, j] = np.corrcoef(y_test_mhls[:, i, j], y_pred_mhls[:, i, j])[0, 1]
                        kge_time[i, j] = GetKGE(y_pred_mhls[:, i, j], y_test_mhls[:, i, j])
                        # pcc_time[i, j] = GetPCC(y_pred_mhls[:, i, j], y_test_mhls[:, i, j])
                        # loss_time[i, j] = NaNMSELoss.fit(cfg,y_pred_mhls[:, i, j], y_test_mhls[:, i, j], lossmse)
                        nse_time[i, j] = GetNSE(y_pred_mhls[:, i, j], y_test_mhls[:, i, j])
                        # bias_time[i, j] = _bias(y_pred_mhls[:, i, j], y_test_mhls[:, i, j])
                        r2_time[i, j] = r2_score(y_test_mhls[:, i, j], y_pred_mhls[:, i, j])
                        rmse_time[i, j] = GetRMSE(y_pred_mhls[:, i, j], y_test_mhls[:, i, j])

            kge.append(kge_time)
            nse.append(nse_time)
            # pcc.append(pcc_time)
            # loss_test.append(loss_time)
            r2.append(r2_time)
            rmse.append(rmse_time)
            # bias.append(bias_time)
            r.append(r_mhls)


        else:
            # 创建极端值标记函数
            def get_extreme_masks(data, percentile=10):
                """
                获取数据中前10%最大值和前10%最小值的格点掩码

                Args:
                    data: 输入数据 (time, lat, lon)
                    percentile: 百分位数，默认10表示前10%

                Returns:
                    high_mask: 高值掩码
                    low_mask: 低值掩码
                """
                print(f"Data shape: {data.shape}")
                print(f"Data contains NaN: {np.isnan(data).any()}")
                print(f"Data range: {np.nanmin(data)} to {np.nanmax(data)}")

                # 检查数据是否为空
                if data.size == 0:
                    print("Warning: Empty data array!")
                    return np.zeros((data.shape[1], data.shape[2]), dtype=bool), np.zeros(
                        (data.shape[1], data.shape[2]), dtype=bool)

                # 计算每个格点的时间平均值，忽略NaN
                with np.errstate(invalid='ignore'):
                    mean_values = np.nanmean(data, axis=0)

                print(f"Mean values shape: {mean_values.shape}")
                print(f"Mean values range: {np.nanmin(mean_values)} to {np.nanmax(mean_values)}")

                # 获取有效格点（非NaN且不是常数）
                valid_mask = ~np.isnan(mean_values)

                # 额外检查：去除值不变的格点
                if data.shape[0] > 1:  # 如果有多个时间步
                    std_values = np.nanstd(data, axis=0)
                    valid_mask = valid_mask & (std_values > 1e-10)  # 标准差大于极小值

                valid_values = mean_values[valid_mask]
                print(f"Valid points count: {len(valid_values)}")

                if len(valid_values) == 0:
                    print("Warning: No valid data points found!")
                    return np.zeros_like(mean_values, dtype=bool), np.zeros_like(mean_values, dtype=bool)

                # 计算阈值
                high_threshold = np.percentile(valid_values, 100 - percentile)
                low_threshold = np.percentile(valid_values, percentile)

                print(f"Thresholds - Low: {low_threshold:.6f}, High: {high_threshold:.6f}")

                # 创建掩码
                high_mask = (mean_values >= high_threshold) & valid_mask
                low_mask = (mean_values <= low_threshold) & valid_mask

                # 确保高值和低值掩码数量合理
                high_count = np.sum(high_mask)
                low_count = np.sum(low_mask)
                expected_count = int(len(valid_values) * percentile / 100)

                print(f"Expected extreme points per category: ~{expected_count}")
                print(f"Actual high extreme points: {high_count}")
                print(f"Actual low extreme points: {low_count}")

                return high_mask, low_mask

            # 为每个变量创建极端值掩码
            extreme_masks = {}
            for num_repeat in range(cfg['num_repeat']):
                print(f"\n=== Processing Variable {cfg['label'][num_repeat]} (index {num_repeat}) ===")

                # 检查数据结构
                current_data = y_test_mhls[num_repeat]
                print(f"Original data shape: {current_data.shape}")
                print(f"Data type: {type(current_data)}")

                # 确保数据是正确的3D格式 (time, lat, lon)
                if len(current_data.shape) != 3:
                    print(f"Error: Expected 3D data, got {len(current_data.shape)}D")
                    continue

                high_mask, low_mask = get_extreme_masks(current_data)
                extreme_masks[num_repeat] = {
                    'high': high_mask,
                    'low': low_mask,
                    'combined': high_mask | low_mask  # 合并高值和低值掩码
                }

                print(f"Variable {cfg['label'][num_repeat]}:")
                print(f"  High extreme points (top 10%): {np.sum(high_mask)}")
                print(f"  Low extreme points (bottom 10%): {np.sum(low_mask)}")
                print(f"  Total extreme points: {np.sum(extreme_masks[num_repeat]['combined'])}")
                print("=" * 60)

            # 常规评估（所有格点）
            for num_repeat in range(cfg['num_repeat']):
                r_mhls = np.full((nlat, nlon), np.nan)
                loss_time = np.full((nlat, nlon), np.nan)
                kge_time = np.full((nlat, nlon), np.nan)
                pcc_time = np.full((nlat, nlon), np.nan)
                nse_time = np.full((nlat, nlon), np.nan)
                r2_time = np.full((nlat, nlon), np.nan)
                rmse_time = np.full((nlat, nlon), np.nan)
                bias_time = np.full((nlat, nlon), np.nan)
                for i in range(nlat):
                    for j in range(nlon):
                        if not (np.isnan(y_test_mhls[num_repeat][:, i, j]).any()):
                            # r_mhls[i, j] = np.corrcoef(y_test_mhls[num_repeat][:, i, j], y_pred_mhls[num_repeat][:, i, j])[0, 1]
                            r2_time[i, j] = r2_score(y_test_mhls[num_repeat][:, i, j], y_pred_mhls[num_repeat][:, i, j])
                            kge_time[i, j] = GetKGE(y_pred_mhls[num_repeat][:, i, j], y_test_mhls[num_repeat][:, i, j])

                kge.append(kge_time)
                r2.append(r2_time)

            # 极端值评估（分别计算高值和低值极端）
            kge_extreme_high = []  # 高极端值（最湿润/最高蒸散发）
            r2_extreme_high = []
            kge_extreme_low = []  # 低极端值（最干燥/最低蒸散发）
            r2_extreme_low = []

            for num_repeat in range(cfg['num_repeat']):
                # 初始化高极端值评估矩阵
                kge_time_extreme_high = np.full((nlat, nlon), np.nan)
                r2_time_extreme_high = np.full((nlat, nlon), np.nan)

                # 初始化低极端值评估矩阵
                kge_time_extreme_low = np.full((nlat, nlon), np.nan)
                r2_time_extreme_low = np.full((nlat, nlon), np.nan)

                # 获取当前变量的极端值掩码
                high_mask = extreme_masks[num_repeat]['high']
                low_mask = extreme_masks[num_repeat]['low']

                # 添加调试计数器
                high_count = 0
                low_count = 0
                high_valid_evaluations = 0
                low_valid_evaluations = 0

                for i in range(nlat):
                    for j in range(nlon):
                        if not (np.isnan(y_test_mhls[num_repeat][:, i, j]).any()):
                            # 高极端值格点评估
                            if high_mask[i, j]:
                                high_count += 1
                                obs_vals = y_test_mhls[num_repeat][:, i, j]
                                pred_vals = y_pred_mhls[num_repeat][:, i, j]

                                # 检查数据质量
                                if not (np.isnan(pred_vals).any() or np.isnan(obs_vals).any()):
                                    r2_val = r2_score(obs_vals, pred_vals)
                                    kge_val = GetKGE(pred_vals, obs_vals)

                                    # 检查异常值
                                    if not (np.isnan(r2_val) or np.isnan(kge_val) or np.isinf(r2_val) or np.isinf(
                                            kge_val)):
                                        r2_time_extreme_high[i, j] = r2_val
                                        kge_time_extreme_high[i, j] = kge_val
                                        high_valid_evaluations += 1
                                    else:
                                        if high_count <= 5:  # 只打印前几个异常情况
                                            print(f"High extreme anomaly at ({i},{j}): R2={r2_val}, KGE={kge_val}")
                                            print(f"  Obs range: {np.min(obs_vals):.6f} - {np.max(obs_vals):.6f}")
                                            print(f"  Pred range: {np.min(pred_vals):.6f} - {np.max(pred_vals):.6f}")

                            # 低极端值格点评估
                            if low_mask[i, j]:
                                low_count += 1
                                obs_vals = y_test_mhls[num_repeat][:, i, j]
                                pred_vals = y_pred_mhls[num_repeat][:, i, j]

                                # 检查数据质量
                                if not (np.isnan(pred_vals).any() or np.isnan(obs_vals).any()):
                                    r2_val = r2_score(obs_vals, pred_vals)
                                    kge_val = GetKGE(pred_vals, obs_vals)

                                    # 检查异常值
                                    if not (np.isnan(r2_val) or np.isnan(kge_val) or np.isinf(r2_val) or np.isinf(
                                            kge_val)):
                                        r2_time_extreme_low[i, j] = r2_val
                                        kge_time_extreme_low[i, j] = kge_val
                                        low_valid_evaluations += 1
                                    else:
                                        if low_count <= 5:  # 只打印前几个异常情况
                                            print(f"Low extreme anomaly at ({i},{j}): R2={r2_val}, KGE={kge_val}")
                                            print(f"  Obs range: {np.min(obs_vals):.6f} - {np.max(obs_vals):.6f}")
                                            print(f"  Pred range: {np.min(pred_vals):.6f} - {np.max(pred_vals):.6f}")
                                            print(
                                                f"  Obs std: {np.std(obs_vals):.6f}, Pred std: {np.std(pred_vals):.6f}")

                print(f"Extreme evaluation summary for {cfg['label'][num_repeat]}:")
                print(f"  High extreme: {high_valid_evaluations}/{high_count} valid evaluations")
                print(f"  Low extreme: {low_valid_evaluations}/{low_count} valid evaluations")

                kge_extreme_high.append(kge_time_extreme_high)
                r2_extreme_high.append(r2_time_extreme_high)
                kge_extreme_low.append(kge_time_extreme_low)
                r2_extreme_low.append(r2_time_extreme_low)

        if cfg['modelname'] in ['LSTM', 'CNN', 'ConvLSTM']:
            y_test_lstm = lon_transform(y_test_mhls)
            mask[-int(mask.shape[0] / 5.4):, :] = 0
            min_map = np.min(y_test_lstm, axis=0)
            max_map = np.max(y_test_lstm, axis=0)
            mask[min_map == max_map] = 0
            r[0] = r[0][mask == 1]
            # kge[0] = kge[0][mask == 1]
            # nse[0] = nse[0][mask == 1]
            # pcc[0] = pcc[0][mask == 1]
            r2[0] = r2[0][mask == 1]
            # rmse[0] = rmse[0][mask == 1]
            # bias[0] = bias[0][mask == 1]
            print('the average r of', cfg['label'], 'model is :', np.nanmedian(r[0]))
            print('the average kge of', cfg['label'], 'model is :', np.nanmedian(kge[0]))
            print('the average nse of', cfg['label'], 'model is :', np.nanmedian(nse[0]))
            print('the average r2 of', cfg['label'], 'model is :', np.nanmedian(r2[0]))
            # print('the average pcc of', cfg['label'], 'model is :', np.nanmedian(pcc[0]))
            # print('the average mse of', cfg['label'], 'model is :', np.nanmedian(loss_test[0]))
            print('the average rmse of', cfg['label'], 'model is :', np.nanmedian(rmse[0]))
            # print('the average bias of', cfg['label'], 'model is :', np.nanmedian(bias[0]))
            np.save(out_path_mhl + 'r_' + cfg['modelname'] + '.npy', r)
            np.save(out_path_mhl + 'r2_' + cfg['modelname'] + '.npy', r2)
            np.save(out_path_mhl + 'kge_' + cfg['modelname'] + '.npy', kge)
            # np.save(out_path_mhl + 'bias_' + cfg['modelname'] + '.npy', bias)
            np.save(out_path_mhl + 'rmse_' + cfg['modelname'] + '.npy', rmse)
            np.save(out_path_mhl + 'nse_' + cfg['modelname'] + '.npy', nse)

        else:
            # 常规评估结果处理
            for i in range(cfg['num_repeat']):
                # 第一行是去掉南极部分，后三行应该是去掉值不发生变化的地区
                y_test_lstm = lon_transform(y_test_mhls[i])
                mask[-int(mask.shape[0] / 5.4):, :] = 0
                min_map = np.min(y_test_lstm, axis=0)
                max_map = np.max(y_test_lstm, axis=0)
                mask[min_map == max_map] = 0

                # 应用掩码到常规评估结果
                kge[i] = kge[i][mask[:, :, i] == 1]
                r2[i] = r2[i][mask[:, :, i] == 1]

                # 应用掩码到极端值评估结果
                kge_extreme_high[i] = kge_extreme_high[i][mask[:, :, i] == 1]
                r2_extreme_high[i] = r2_extreme_high[i][mask[:, :, i] == 1]
                kge_extreme_low[i] = kge_extreme_low[i][mask[:, :, i] == 1]
                r2_extreme_low[i] = r2_extreme_low[i][mask[:, :, i] == 1]

                # 过滤异常值函数
                def filter_outliers(data):
                    """过滤掉NaN、无穷大和异常的负值"""
                    filtered = data[np.isfinite(data) & (data > -2) & (data < 10)]
                    return filtered

                # 极端值评估结果
                print(f'=== Extreme Values Evaluation for {cfg["label"][i]} ===')

                # 高极端值处理
                kge_high_original = kge_extreme_high[i][~np.isnan(kge_extreme_high[i])]
                r2_high_original = r2_extreme_high[i][~np.isnan(r2_extreme_high[i])]
                kge_high_filtered = filter_outliers(kge_extreme_high[i])
                r2_high_filtered = filter_outliers(r2_extreme_high[i])

                print(
                    f'High extreme (top 10%): {len(kge_high_original)} -> {len(kge_high_filtered)} points after filtering')
                print(f'  KGE: {np.nanmedian(kge_high_filtered):.4f}')
                print(f'  R²:  {np.nanmedian(r2_high_filtered):.4f}')

                # 低极端值处理
                kge_low_original = kge_extreme_low[i][~np.isnan(kge_extreme_low[i])]
                r2_low_original = r2_extreme_low[i][~np.isnan(r2_extreme_low[i])]
                kge_low_filtered = filter_outliers(kge_extreme_low[i])
                r2_low_filtered = filter_outliers(r2_extreme_low[i])

                print(
                    f'Low extreme (bottom 10%): {len(kge_low_original)} -> {len(kge_low_filtered)} points after filtering')
                print(f'  KGE: {np.nanmedian(kge_low_filtered):.4f}')
                print(f'  R²:  {np.nanmedian(r2_low_filtered):.4f}')

                # 保存过滤后的值用于后续计算
                kge_high_median = np.nanmedian(kge_high_filtered) if len(kge_high_filtered) > 0 else np.nan
                r2_high_median = np.nanmedian(r2_high_filtered) if len(r2_high_filtered) > 0 else np.nan
                kge_low_median = np.nanmedian(kge_low_filtered) if len(kge_low_filtered) > 0 else np.nan
                r2_low_median = np.nanmedian(r2_low_filtered) if len(r2_low_filtered) > 0 else np.nan

                print('=' * 50)
                # print('the average pcc of', cfg['label'][i], 'model is :', np.nanmedian(pcc[i]))
                # print('the average mse of', cfg['label'][i], 'model is :', np.nanmedian(loss_test[i]))
                # print('the average rmse of', cfg['label'][i], 'model is :', np.nanmedian(rmse[i]))
                # print('the average bias of', cfg['label'][i], 'model is :', np.nanmedian(bias[i]))
            # 保存常规评估结果和极端值评估结果
            if cfg['label'][-1] in ['total_evaporation']:
                # 保存常规评估结果
                with open(out_path_mhl + 'r2_' + cfg['modelname'] + '.npy', 'wb') as f:
                    pickle.dump(r2, f)
                with open(out_path_mhl + 'kge_' + cfg['modelname'] + '.npy', 'wb') as f:
                    pickle.dump(kge, f)

                # 保存高极端值评估结果
                with open(out_path_mhl + 'r2_extreme_high_' + cfg['modelname'] + '.npy', 'wb') as f:
                    pickle.dump(r2_extreme_high, f)
                with open(out_path_mhl + 'kge_extreme_high_' + cfg['modelname'] + '.npy', 'wb') as f:
                    pickle.dump(kge_extreme_high, f)

                # 保存低极端值评估结果
                with open(out_path_mhl + 'r2_extreme_low_' + cfg['modelname'] + '.npy', 'wb') as f:
                    pickle.dump(r2_extreme_low, f)
                with open(out_path_mhl + 'kge_extreme_low_' + cfg['modelname'] + '.npy', 'wb') as f:
                    pickle.dump(kge_extreme_low, f)

                # 保存极端值掩码
                with open(out_path_mhl + 'extreme_masks_' + cfg['modelname'] + '.npy', 'wb') as f:
                    pickle.dump(extreme_masks, f)

            else:
                # 保存常规评估结果
                np.save(out_path_mhl + 'r2_' + cfg['modelname'] + '.npy', r2)
                np.save(out_path_mhl + 'kge_' + cfg['modelname'] + '.npy', kge)

                # 保存高极端值评估结果
                np.save(out_path_mhl + 'r2_extreme_high_' + cfg['modelname'] + '.npy', r2_extreme_high)
                np.save(out_path_mhl + 'kge_extreme_high_' + cfg['modelname'] + '.npy', kge_extreme_high)

                # 保存低极端值评估结果
                np.save(out_path_mhl + 'r2_extreme_low_' + cfg['modelname'] + '.npy', r2_extreme_low)
                np.save(out_path_mhl + 'kge_extreme_low_' + cfg['modelname'] + '.npy', kge_extreme_low)

                # 保存极端值掩码
                np.save(out_path_mhl + 'extreme_masks_' + cfg['modelname'] + '.npy', extreme_masks)

        print('postprocess ove, please go on')
    # ----------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    cfg = get_args()
    postprocess(cfg)







