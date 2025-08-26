%% 感知机实验完整实现 - 南开大学机器学习实验课

%% 1. 数据生成（基于提供的代码）
clear all; close all; clc;

% 训练数据生成（每类100个）
n_train = 100;        % 每类样本量
center1 = [1, 1];     % 第一类数据中心
center2 = [3, 4];     % 第二类数据中心

% 生成训练数据矩阵（200x2）
X_train = zeros(2*n_train, 2);
Y_train = zeros(2*n_train, 1);

% 第一类数据（标签=1）
X_train(1:n_train, :) = ones(n_train, 1)*center1 + randn(n_train, 2);
Y_train(1:n_train) = 1;

% 第二类数据（标签=-1）
X_train(n_train+1:2*n_train, :) = ones(n_train, 1)*center2 + randn(n_train, 2);
Y_train(n_train+1:2*n_train) = -1;

% 测试数据生成（每类10个）
n_test = 10;
X_test = zeros(2*n_test, 2);
Y_test = zeros(2*n_test, 1);

% 第一类测试数据
X_test(1:n_test, :) = ones(n_test, 1)*center1 + randn(n_test, 2);
Y_test(1:n_test) = 1;

% 第二类测试数据
X_test(n_test+1:2*n_test, :) = ones(n_test, 1)*center2 + randn(n_test, 2);
Y_test(n_test+1:2*n_test) = -1;

% 可视化训练数据
figure(1)
set(gcf, 'Position', [100, 100, 700, 600], 'color', 'w')
set(gca, 'Fontsize', 12)
plot(X_train(1:n_train, 1), X_train(1:n_train, 2), 'ro', 'LineWidth', 1, 'MarkerSize', 6)
hold on;
plot(X_train(n_train+1:2*n_train, 1), X_train(n_train+1:2*n_train, 2), 'b*', 'LineWidth', 1, 'MarkerSize', 6)
xlabel('x axis');
ylabel('y axis');
title('训练数据分布');
legend('class 1 (train)', 'class 2 (train)');
grid on;

%% 2. 感知机实现（手动实现随机梯度下降）
% 初始化参数
w = zeros(2, 1);  % 权重向量 (2x1)
b = 0;            % 偏置项
eta = 0.1;        % 学习率
max_epochs = 100; % 最大迭代次数
converged = false;% 收敛标志

% 训练过程记录（用于可视化）
training_history = struct('w', cell(max_epochs,1), 'b', cell(max_epochs,1));

% 随机梯度下降
for epoch = 1:max_epochs
    misclassified = 0;
    
    % 保存当前参数（用于历史记录）
    training_history(epoch).w = w;
    training_history(epoch).b = b;
    
    % 遍历所有样本
    for i = 1:length(Y_train)
        % 计算预测值
        prediction = sign(w' * X_train(i, :)' + b);
        
        % 检查是否误分类
        if Y_train(i) * (w' * X_train(i, :)' + b) <= 0
            % 更新权重和偏置
            w = w + eta * Y_train(i) * X_train(i, :)';
            b = b + eta * Y_train(i);
            misclassified = misclassified + 1;
        end
    end
    
    % 如果没有误分类点，提前终止
    if misclassified == 0
        converged = true;
        fprintf('在第 %d 轮迭代后收敛\n', epoch);
        break;
    end
    
    % 每10轮显示进度
    if mod(epoch, 10) == 0
        fprintf('第 %d 轮迭代，误分类点数: %d\n', epoch, misclassified);
    end
end

% 最终训练结果
fprintf('训练完成! 最终参数: w = [%.4f, %.4f], b = %.4f\n', w(1), w(2), b);

%% 3. 可视化分类边界
% 计算分类边界
x_range = [min(X_train(:,1))-1, max(X_train(:,1))+1];
x1 = linspace(x_range(1), x_range(2), 1000);
y1 = (-b - w(1)*x1) / w(2);  % w1*x + w2*y + b = 0

% 绘制分类结果
figure(2)
set(gcf, 'Position', [100, 100, 900, 700], 'color', 'w')
set(gca, 'Fontsize', 12)

% 绘制训练数据
plot(X_train(1:n_train, 1), X_train(1:n_train, 2), 'ro', 'LineWidth', 1, 'MarkerSize', 6)
hold on;
plot(X_train(n_train+1:2*n_train, 1), X_train(n_train+1:2*n_train, 2), 'b*', 'LineWidth', 1, 'MarkerSize', 6)

% 绘制分类边界
plot(x1, y1, 'k-', 'LineWidth', 2)
xlabel('x axis');
ylabel('y axis');
title('感知机分类结果');
legend('class 1 (train)', 'class 2 (train)', '决策边界');
grid on;
axis equal;

%% 4. 测试集评估
% 预测测试集
correct = 0;
predictions = zeros(size(Y_test));

for i = 1:length(Y_test)
    % 计算预测值
    pred_val = sign(w' * X_test(i, :)' + b);
    predictions(i) = pred_val;
    
    % 检查是否正确分类
    if pred_val == Y_test(i)
        correct = correct + 1;
    end
end

% 计算准确率
accuracy = correct / length(Y_test) * 100;
fprintf('测试准确率: %.2f%% (%d/%d)\n', accuracy, correct, length(Y_test));

% 可视化测试结果
figure(3)
set(gcf, 'Position', [100, 100, 900, 700], 'color', 'w')
set(gca, 'Fontsize', 12)

% 绘制训练数据（浅色）
plot(X_train(1:n_train, 1), X_train(1:n_train, 2), 'ro', 'MarkerFaceColor', [1 0.8 0.8], 'MarkerSize', 4)
hold on;
plot(X_train(n_train+1:2*n_train, 1), X_train(n_train+1:2*n_train, 2), 'b*', 'MarkerEdgeColor', [0.7 0.7 1], 'MarkerSize', 6)

% 绘制测试数据（深色）
plot(X_test(1:n_test, 1), X_test(1:n_test, 2), 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 8)
plot(X_test(n_test+1:2*n_test, 1), X_test(n_test+1:2*n_test, 2), 'b*', 'MarkerSize', 10)

% 绘制分类边界
plot(x1, y1, 'k-', 'LineWidth', 2)

% 标记错误分类点
misclassified_idx = find(predictions ~= Y_test);
if ~isempty(misclassified_idx)
    plot(X_test(misclassified_idx, 1), X_test(misclassified_idx, 2), 'ks', 'MarkerSize', 12, 'LineWidth', 2)
    legend_str = {'class 1 (train)', 'class 2 (train)', 'class 1 (test)', 'class 2 (test)', '决策边界', '错误分类'};
else
    legend_str = {'class 1 (train)', 'class 2 (train)', 'class 1 (test)', 'class 2 (test)', '决策边界'};
end

xlabel('x axis');
ylabel('y axis');
title(sprintf('测试结果 (准确率: %.2f%%)', accuracy));
legend(legend_str);
grid on;
axis equal;

%% 5. 附加题1：不同迭代次数的决策边界变化（示例）
if exist('training_history', 'var')
    figure(4)
    set(gcf, 'Position', [100, 100, 1200, 900], 'color', 'w')
    set(gca, 'Fontsize', 10)
    
    % 选择关键迭代点
    plot_epochs = [1, 5, 10, 20, 50, min(epoch, max_epochs)];
    colors = lines(length(plot_epochs));
    
    % 绘制训练数据
    plot(X_train(1:n_train, 1), X_train(1:n_train, 2), 'ro', 'MarkerSize', 4, 'MarkerFaceColor', [1 0.8 0.8])
    hold on;
    plot(X_train(n_train+1:2*n_train, 1), X_train(n_train+1:2*n_train, 2), 'b*', 'MarkerEdgeColor', [0.7 0.7 1], 'MarkerSize', 6)
    
    % 绘制不同迭代次数的决策边界
    legend_str = {'class 1', 'class 2'};
    for i = 1:length(plot_epochs)
        e = plot_epochs(i);
        w_hist = training_history(e).w;
        b_hist = training_history(e).b;
        
        % 计算分类边界
        y_hist = (-b_hist - w_hist(1)*x1) / w_hist(2);
        
        % 绘制边界
        plot(x1, y_hist, '-', 'LineWidth', 1.5, 'Color', colors(i, :))
        legend_str{end+1} = sprintf('迭代 %d', e);
    end
    
    xlabel('x axis');
    ylabel('y axis');
    title('不同迭代次数的决策边界变化');
    legend(legend_str, 'Location', 'eastoutside');
    grid on;
    axis equal;
end

%% 6. 实验报告生成（结果输出）
fprintf('\n===== 实验报告摘要 =====\n');
fprintf('实验名称: 感知机模型实现\n');
fprintf('训练数据: 每类100个样本 (共200个)\n');
fprintf('测试数据: 每类10个样本 (共20个)\n');
fprintf('学习率(η): %.2f\n', eta);
fprintf('最大迭代次数: %d\n', max_epochs);
fprintf('实际迭代次数: %d\n', epoch);
fprintf('收敛状态: %s\n', converged);
fprintf('最终权重 w: [%.4f, %.4f]\n', w(1), w(2));
fprintf('最终偏置 b: %.4f\n', b);
fprintf('测试准确率: %.2f%%\n', accuracy);
fprintf('========================\n');