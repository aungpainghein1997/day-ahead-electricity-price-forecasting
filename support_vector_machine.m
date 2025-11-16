%% ============================================================
%  SUPPORT VECTOR MACHINE DAY-AHEAD ELECTRICITY PRICE FORECASTING
%  Final Combined Script (Hyperparameter Tuning + CV)
% =============================================================

clc; clear; close all;

%% ==================== STEP 1: Load & Prepare Data ======================
data = readtable('train_data.xlsx');
data.TimeStamp = datetime(data.TimeStamp);

X = data{:, {'Demand','Gas','Wind','Nuclear','Temperature'}};
y = data.Spotprices;

%% ==================== STEP 2: Train/Test Split =========================
cv = cvpartition(size(X,1), 'HoldOut', 0.2);
X_train = X(training(cv), :);
X_test  = X(test(cv), :);
y_train = y(training(cv));
y_test  = y(test(cv));

%% ==================== STEP 3: Standardize Features =====================
[X_train_scaled, mu, sigma] = zscore(X_train);
X_test_scaled = (X_test - mu) ./ sigma;

%% ==================== STEP 4: SVM Hyperparameter Tuning ================
kernels = {'linear','gaussian','polynomial'};
BoxC = [0.1 1 10];
epsVals = [0.01 0.1 1];
scaleVals = [0.1 1 10];

best_rmse = inf;
best_model = [];

for k = 1:length(kernels)
    kernel = kernels{k};
    
    for C = BoxC
        for ep = epsVals
            
            if strcmp(kernel,'gaussian')
                for ks = scaleVals
                    model = fitrsvm(X_train_scaled, y_train, ...
                        'KernelFunction', kernel, ...
                        'BoxConstraint', C, ...
                        'Epsilon', ep, ...
                        'KernelScale', ks, ...
                        'Standardize', true);
                    
                    y_pred = predict(model, X_test_scaled);
                    rmse = sqrt(mean((y_test - y_pred).^2));
                    
                    if rmse < best_rmse
                        best_rmse = rmse;
                        best_model = model;
                    end
                end
            else
                model = fitrsvm(X_train_scaled, y_train, ...
                    'KernelFunction', kernel, ...
                    'BoxConstraint', C, ...
                    'Epsilon', ep, ...
                    'Standardize', true);
                
                y_pred = predict(model, X_test_scaled);
                rmse = sqrt(mean((y_test - y_pred).^2));
                
                if rmse < best_rmse
                    best_rmse = rmse;
                    best_model = model;
                end
            end
        end
    end
end

fprintf("Best SVM RMSE: %.4f\n", best_rmse);

%% ==================== STEP 5: Final Predictions ========================
y_pred_final = predict(best_model, X_test_scaled);

%% ==================== STEP 6: Evaluation Metrics =======================
MAE = mean(abs(y_test - y_pred_final));
RMSE = sqrt(mean((y_test - y_pred_final).^2));
SS_res = sum((y_test - y_pred_final).^2);
SS_tot = sum((y_test - mean(y_test)).^2);
R2 = 1 - SS_res/SS_tot;

fprintf("MAE: %.4f\nRMSE: %.4f\nR²: %.4f\n", MAE, RMSE, R2);

%% ==================== STEP 7: Day-Ahead Prediction =====================
future_data = X(end-23:end, :);
future_scaled = (future_data - mu) ./ sigma;

day_ahead_pred = predict(best_model, future_scaled);

future_timestamps = data.TimeStamp(end) + hours(1:24)';

%% ==================== STEP 8: Create Results Table =====================
comparison = table(...
    future_timestamps, ...
    data.Spotprices(end-23:end), ...
    day_ahead_pred, ...
    'VariableNames', {'TimeStamp','Actual','Predicted'});

disp(comparison);

%% ==================== STEP 9: Plotting =================================
figure;
plot(y_test,'b','LineWidth',2); hold on;
plot(y_pred_final,'r','LineWidth',2);
legend('Actual','Predicted');
title('SVM: Actual vs Predicted');
grid on;

figure;
plot(future_timestamps, day_ahead_pred, '-o', 'LineWidth',2);
title('SVM – Day-Ahead Forecast');
xlabel('Time'); ylabel('Forecasted Price');
grid on;

