%% ERα活性预测的分子描述符筛选
% 本代码分析分子描述符与ERα活性的关系，筛选最重要的20个分子描述符
% 处理流程：预处理→特征筛选→相关性验证

%% 1. 数据加载与预处理
% 读取活性数据
disp('正在加载数据...');
activity_data = readtable('ERα_activity.xlsx', 'Sheet', 'training');
descriptor_data = readtable('Molecular_Descriptor.xlsx', 'Sheet', 'training');

% 显示表格的列名以便检查
disp('活性数据表的列名:');
disp(activity_data.Properties.VariableNames);
disp('分子描述符表的列名:');
disp(descriptor_data.Properties.VariableNames(1:5)); % 只显示前5个，避免输出过多

% 提取SMILES、活性值和分子描述符 - 使用列名而不是列索引
smiles_activity = activity_data.SMILES;  
ic50 = activity_data.IC50_nM;            
pic50 = activity_data.pIC50;           
smiles_descriptor = descriptor_data.SMILES;  
descriptors = descriptor_data(:, 2:end);    
descriptor_names = descriptor_data.Properties.VariableNames(2:end);

% 确保两个数据集中的化合物顺序一致
if ~isequal(smiles_activity, smiles_descriptor)
    warning('两个Excel文件中的化合物顺序可能不一致，请检查数据!');
    % 显示一些样本进行比较
    disp('活性数据的前5个SMILES:');
    disp(smiles_activity(1:5));
    disp('描述符数据的前5个SMILES:');
    disp(smiles_descriptor(1:5));
end

% 处理缺失值
disp('处理缺失值...');
nan_cols = sum(ismissing(descriptors));
disp(['包含缺失值的特征数量: ', num2str(sum(nan_cols > 0))]);

% 移除包含超过5%缺失值的特征
threshold = 0.05 * height(descriptors);
cols_to_remove = nan_cols > threshold;
descriptors = descriptors(:, ~cols_to_remove);
descriptor_names = descriptor_names(~cols_to_remove);
disp(['移除超过5%缺失值的特征后，剩余特征数量: ', num2str(width(descriptors))]);

% 对剩余的缺失值进行填充（使用中位数）
for i = 1:width(descriptors)
    col = descriptors{:, i};
    if any(ismissing(col))
        col_median = median(col, 'omitnan');
        col(ismissing(col)) = col_median;
        descriptors{:, i} = col;
    end
end

% 将表格转换为矩阵以加速计算
X = table2array(descriptors);
y = pic50;

% 数据标准化
disp('数据标准化...');
[X_scaled, mu, sigma] = zscore(X);

% 移除方差接近于零的特征
near_zero_var = var(X_scaled) < 0.01;
X_scaled = X_scaled(:, ~near_zero_var);
descriptor_names = descriptor_names(~near_zero_var);
disp(['移除低方差特征后，剩余特征数量: ', num2str(size(X_scaled, 2))]);

% 移除高度相关的特征 (相关系数 > 0.9)
disp('移除高度相关特征...');
R = corrcoef(X_scaled);
R = triu(R, 1); % 只保留上三角矩阵（不包括对角线）
[rows, cols] = find(abs(R) > 0.9);
to_remove = false(1, size(X_scaled, 2));

% 对每对高相关特征，保留与目标变量相关性更高的特征
for i = 1:length(rows)
    corr1 = abs(corr(X_scaled(:, rows(i)), y));
    corr2 = abs(corr(X_scaled(:, cols(i)), y));
    if corr1 < corr2
        to_remove(rows(i)) = true;
    else
        to_remove(cols(i)) = true;
    end
end

X_scaled = X_scaled(:, ~to_remove);
descriptor_names = descriptor_names(~to_remove);
disp(['移除高相关特征后，剩余特征数量: ', num2str(size(X_scaled, 2))]);

%% 2. 特征选择
% 使用多种方法进行特征选择并整合结果

% 2.1 LASSO回归进行特征选择
disp('使用LASSO回归进行特征选择...');
[B, FitInfo] = lasso(X_scaled, y, 'CV', 10);
idxLambdaMinMSE = FitInfo.IndexMinMSE;
lasso_coef = B(:, idxLambdaMinMSE);
lasso_importance = abs(lasso_coef);

% 确保所有重要性向量具有相同的长度
disp(['LASSO系数长度: ', num2str(length(lasso_importance))]);

% 修复: 替换LASSO路径图为手动绘制的系数图
figure('Visible', 'on');
plot(log(FitInfo.Lambda), B, 'LineWidth', 1);
xlabel('Log Lambda');
ylabel('系数');
title('LASSO系数路径图');
grid on;
saveas(gcf, 'lasso_path.png');

% 绘制LASSO系数柱状图 (更加直观)
figure('Visible', 'on');
bar(lasso_importance);
xlabel('特征索引');
ylabel('LASSO系数绝对值');
title('LASSO特征重要性');
saveas(gcf, 'lasso_importance.png');

% 2.2 使用随机森林评估特征重要性
disp('使用随机森林评估特征重要性...');
rng(1); % 设置随机种子以确保结果可重复
nTrees = 500;
% 修复: 添加 'OOBPredictorImportance' 参数设置为 'on'
rf_model = TreeBagger(nTrees, X_scaled, y, 'Method', 'regression', 'OOBPrediction', 'on', 'OOBPredictorImportance', 'on', 'PredictorSelection', 'curvature');
rf_importance = rf_model.OOBPermutedPredictorDeltaError;

disp(['随机森林特征重要性长度: ', num2str(length(rf_importance))]);

% 绘制随机森林特征重要性
figure('Visible', 'on');
bar(rf_importance);
xlabel('特征索引');
ylabel('特征重要性');
title('随机森林特征重要性');
saveas(gcf, 'rf_importance.png');

% 显示随机森林OOB误差
oobError = rf_model.OOBPermutedPredictorDeltaError;
figure('Visible', 'on');
plot(oobError);
xlabel('特征索引');
ylabel('OOB误差');
title('随机森林OOB误差');
saveas(gcf, 'rf_oob_error.png');

% 2.3 相关性分析
disp('计算特征与目标变量的相关性...');
corr_with_target = zeros(size(X_scaled, 2), 1);
for i = 1:size(X_scaled, 2)
    corr_with_target(i) = abs(corr(X_scaled(:, i), y));
end

disp(['相关性向量长度: ', num2str(length(corr_with_target))]);
disp(['描述符名称长度: ', num2str(length(descriptor_names))]);

% 绘制相关性直方图
figure('Visible', 'on');
histogram(corr_with_target, 20);
xlabel('与目标变量的绝对相关系数');
ylabel('频率');
title('特征与目标变量相关性分布');
saveas(gcf, 'correlation_histogram.png');

% 2.4 整合多种方法的结果
% 确保三个向量长度相同
if length(lasso_importance) ~= length(rf_importance) || length(lasso_importance) ~= length(corr_with_target)
    disp('警告: 三种方法生成的特征重要性向量长度不一致!');
    disp(['LASSO: ', num2str(length(lasso_importance)), ' RF: ', num2str(length(rf_importance)), ' 相关性: ', num2str(length(corr_with_target))]);
    
    % 使用最短的长度
    min_length = min([length(lasso_importance), length(rf_importance), length(corr_with_target)]);
    lasso_importance = lasso_importance(1:min_length);
    rf_importance = rf_importance(1:min_length);
    corr_with_target = corr_with_target(1:min_length);
    descriptor_names = descriptor_names(1:min_length);
    
    disp(['使用最短的长度: ', num2str(min_length)]);
end

% 标准化各种重要性分数
lasso_norm = lasso_importance / max(lasso_importance + eps);
rf_norm = rf_importance / max(rf_importance + eps);
corr_norm = corr_with_target / max(corr_with_target + eps);

% 计算综合重要性分数
combined_importance = (lasso_norm + rf_norm + corr_norm) / 3;

% 对特征按综合重要性进行排序
[sorted_importance, sorted_idx] = sort(combined_importance, 'descend');

% 确保不超出索引范围
top_count = min(20, length(sorted_importance));
disp(['选择的特征数量: ', num2str(top_count)]);

top_features_idx = sorted_idx(1:top_count);
top_features = descriptor_names(top_features_idx);
top_importance = combined_importance(top_features_idx);

% 显示数组维度，以便调试
disp(['top_features 维度: ', num2str(size(top_features))]);
disp(['top_importance 维度: ', num2str(size(top_importance))]);

% 确保两个数组维度匹配 - 都转换为列向量
top_features = top_features(:);  % 确保是列向量
top_importance = top_importance(:);  % 确保是列向量

disp(['调整后 top_features 维度: ', num2str(size(top_features))]);
disp(['调整后 top_importance 维度: ', num2str(size(top_importance))]);

%% 3. 相关性验证与可视化
% 3.1 输出前20个最重要特征
disp('前20个最重要的分子描述符:');
result_table = table(top_features, top_importance, 'VariableNames', {'分子描述符', '重要性得分'});
disp(result_table);

% 3.2 绘制前20个特征的重要性条形图
figure('Visible', 'on', 'Position', [100, 100, 800, 600]);  % 设置更大的图形窗口
barh(top_importance);
yticks(1:length(top_importance));
yticklabels(top_features);
xlabel('特征重要性得分');
title('最重要的分子描述符');
set(gca, 'FontSize', 10);
grid on;
box on;
saveas(gcf, 'top_features.png');

% 重要性散点图 - 另一种可视化方式
figure('Visible', 'on');
scatter(1:length(top_importance), top_importance, 100, 'filled');
xticks(1:length(top_importance));
xticklabels(top_features);
xtickangle(45);
xlabel('分子描述符');
ylabel('重要性得分');
title('重要分子描述符排序');
grid on;
saveas(gcf, 'top_features_scatter.png');

% 3.3 前20个特征的相关性热图
X_top = X_scaled(:, top_features_idx);
corr_matrix = corrcoef(X_top);

figure('Visible', 'on', 'Position', [100, 100, 800, 800]);
hmap = heatmap(top_features, top_features, corr_matrix);
hmap.Title = '重要特征的相关性热图';
hmap.XLabel = '分子描述符';
hmap.YLabel = '分子描述符';
hmap.Colormap = jet;
saveas(gcf, 'top_features_correlation.png');

% 3.4 验证所选特征的预测性能
disp('验证所选特征的预测性能...');
rng(1); % 设置随机种子
cv = cvpartition(length(y), 'KFold', 5);

% 使用前20个特征和所有特征的模型进行比较
rmse_top = zeros(cv.NumTestSets, 1);
rmse_all = zeros(cv.NumTestSets, 1);
r2_top = zeros(cv.NumTestSets, 1);
r2_all = zeros(cv.NumTestSets, 1);

for i = 1:cv.NumTestSets
    % 获取训练和测试集索引
    trainIdx = cv.training(i);
    testIdx = cv.test(i);
    
    % 使用前20个特征训练模型
    X_train_top = X_scaled(trainIdx, top_features_idx);
    X_test_top = X_scaled(testIdx, top_features_idx);
    y_train = y(trainIdx);
    y_test = y(testIdx);
    
    model_top = fitrsvm(X_train_top, y_train, 'KernelFunction', 'rbf', 'Standardize', true);
    y_pred_top = predict(model_top, X_test_top);
    rmse_top(i) = sqrt(mean((y_pred_top - y_test).^2));
    r2_top(i) = 1 - sum((y_test - y_pred_top).^2) / sum((y_test - mean(y_test)).^2);
    
    % 使用所有特征训练模型
    X_train_all = X_scaled(trainIdx, :);
    X_test_all = X_scaled(testIdx, :);
    
    model_all = fitrsvm(X_train_all, y_train, 'KernelFunction', 'rbf', 'Standardize', true);
    y_pred_all = predict(model_all, X_test_all);
    rmse_all(i) = sqrt(mean((y_pred_all - y_test).^2));
    r2_all(i) = 1 - sum((y_test - y_pred_all).^2) / sum((y_test - mean(y_test)).^2);
end

% 显示预测性能比较
disp(['使用选出的重要特征的平均RMSE: ', num2str(mean(rmse_top))]);
disp(['使用所有特征的平均RMSE: ', num2str(mean(rmse_all))]);
disp(['使用选出的重要特征的平均R²: ', num2str(mean(r2_top))]);
disp(['使用所有特征的平均R²: ', num2str(mean(r2_all))]);

% 绘制预测性能比较
figure('Visible', 'on');
metrics = [mean(rmse_top), mean(rmse_all); mean(r2_top), mean(r2_all)];
bar(metrics);
xticklabels({'RMSE', 'R²'});
legend('重要特征', '所有特征', 'Location', 'best');
ylabel('评价指标值');
title('特征选择前后的预测性能比较');
grid on;
saveas(gcf, 'prediction_performance.png');

% 最后一个特征与目标变量关系的散点图 - 使用最重要的特征
top_feature_idx = top_features_idx(1);
figure('Visible', 'on');
scatter(X_scaled(:, top_feature_idx), y, 'filled');
hold on;
% 添加趋势线
p = polyfit(X_scaled(:, top_feature_idx), y, 1);
x_trend = linspace(min(X_scaled(:, top_feature_idx)), max(X_scaled(:, top_feature_idx)), 100);
y_trend = polyval(p, x_trend);
plot(x_trend, y_trend, 'r-', 'LineWidth', 2);
xlabel(['特征: ', descriptor_names{top_feature_idx}]);
ylabel('pIC50值');
title(['最重要特征与pIC50的关系 (相关系数: ', num2str(corr(X_scaled(:, top_feature_idx), y)), ')']);
grid on;
saveas(gcf, 'top_feature_relationship.png');

% 保存筛选结果到Excel文件
writetable(result_table, 'top_descriptors.xlsx');
disp('分析完成! 重要分子描述符已保存至top_descriptors.xlsx');

% 显示最终的重要性表
disp('最终筛选出的重要分子描述符 (按重要性排序):');
disp(result_table);