%% ============================================================
%  RANDOM FOREST DAY-AHEAD ELECTRICITY PRICE FORECASTING
%  Final Combined Script (Lag Features + Hyperparameter Tuning)
% =============================================================

clc; clear; close all;

%% ==================== STEP 1: Load & Prepare Data ======================
data = readtable('train_data.xlsx');

% Convert TimeStamp to datetime
data.TimeStamp = datetime(data.TimeStamp);

% Create time features
data.Hour = hour(data.TimeStamp);
data.Day = day(data.TimeStamp);
data.Month = month(data.TimeStamp);
data.Weekday = weekday(data.TimeStamp);

%% ==================== STEP 2: Create Lag Features (1–24 hours) =========
for i = 1:24
    data.(sprintf('Demand_lag%d', i))       = circshift(data.Demand, i);
    data.(sprintf('Gas_lag%d', i))          = circshift(data.Gas, i);
    data.(sprintf('Wind_lag%d', i))         = circshift(data.Wind, i);
    data.(sprintf('Nuclear_lag%d', i))      = circshift(data.Nuclear, i);
    data.(sprintf('Temperature_lag%d', i))  = circshift(data.Temperature, i);
    data.(sprintf('Spotprices_lag%d', i))   = circshift(data.Spotprices, i);
end

% Remove rows with missing lag values
data = rmmissing(data);

%% ==================== STEP 3: Build Feature Set ========================
predictorNames = [...
    {'Demand','Gas','Wind','Nuclear','Temperature','Hour','Day','Month','Weekday'} ...
    arrayfun(@(i) sprintf('Demand_lag%d', i), 1:24, 'UniformOutput', false) ...
    arrayfun(@(i) sprintf('Gas_lag%d', i), 1:24, 'UniformOutput', false) ...
    arrayfun(@(i) sprintf('Wind_lag%d', i), 1:24, 'UniformOutput', false) ...
    arrayfun(@(i) sprintf('Nuclear_lag%d', i), 1:24, 'UniformOutput', false) ...
    arrayfun(@(i) sprintf('Temperature_lag%d', i), 1:24, 'UniformOutput', false) ...
    arrayfun(@(i) sprintf('Spotprices_lag%d', i), 1:24, 'UniformOutput', false) ...
];

X = data{:, predictorNames};
y = data.Spotprices;

%% ==================== STEP 4: Split the Data (80/20) ===================
cv = cvpartition(size(X,1), 'HoldOut', 0.2);
X_train = X(training(cv), :);
y_train = y(training(cv));
X_test  = X(test(cv), :);
y_test  = y(test(cv));

%% ==================== STEP 5: Hyperparameter Tuning ====================
numTrees = [50 100 150];
maxSplits = [5 10 15];
minLeaf = [1 5 10];

best_rmse = inf;
best_model = [];

for nt = numTrees
    for ms = maxSplits
        for ml = minLeaf
            
            model = TreeBagger(nt, X_train, y_train, ...
                'Method', 'regression', ...
                'MaxNumSplits', ms, ...
                'MinLeafSize', ml);
            
            y_pred = predict(model, X_test);
            rmse = sqrt(mean((y_test - y_pred).^2));
            
            if rmse < best_rmse
                best_rmse = rmse;
                best_model = model;
            end
        end
    end
end

fprintf("Best RF RMSE: %.4f\n", best_rmse);

%% ==================== STEP 6: Final Predictions ========================
y_pred_final = predict(best_model, X_test);

%% ==================== STEP 7: Evaluation Metrics =======================
MAE = mean(abs(y_test - y_pred_final));
MSE = mean((y_test - y_pred_final).^2);
RMSE = sqrt(MSE);

SS_res = sum((y_test - y_pred_final).^2);
SS_tot = sum((y_test - mean(y_test)).^2);
R2 = 1 - SS_res/SS_tot;

fprintf("MAE: %.4f\nRMSE: %.4f\nR²: %.4f\n", MAE, RMSE, R2);

%% ==================== STEP 8: Predict Next 24 Hours ======================
X_last = X(end-23:end, :);
y_pred_day_ahead = predict(best_model, X_last);

timestamps_next24 = data.TimeStamp(end) + hours(1:24)';

%% ==================== STEP 9: Create Output Table =======================
comparison = table(...
    timestamps_next24, ...
    data.Spotprices(end-23:end), ...
    y_pred_day_ahead, ...
    'VariableNames', {'TimeStamp','Actual','Predicted'});

disp(comparison);

%% ==================== STEP 10: Plot Results =============================
figure;
plot(y_test, 'b', 'LineWidth', 2); hold on;
plot(y_pred_final, 'r', 'LineWidth', 2);
legend('Actual','Predicted');
grid on; title('RF: Actual vs Predicted');

figure;
plot(timestamps_next24, y_pred_day_ahead, '-o', 'LineWidth', 2);
xlabel('Time'); ylabel('Forecasted Price');
title('Random Forest – Day-Ahead Forecast');
grid on;

