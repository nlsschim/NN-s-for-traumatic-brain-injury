%% preprocess and load data

% Specify filepath for csv data
filepath = '/Users/nlsschim/Documents/LAB/ML_TBI/DoD012_TPIHarmonicsTable (1).csv';
feature_data = import_features(filepath); % extract features from csv
target_data = import_target(filepath); % extract targets from csv

func = @(var) round(var,0); % set rounding fxn
target_data = varfun(func, target_data); % round target data to be binary

writetable(target_data, 'target_data.csv', 'Delimiter',',','QuoteStrings',true);
writetable(feature_data, 'feature_data.csv', 'Delimiter', ',', 'QuoteStrings', true);

target_data = table2array(target_data)';
feature_data = table2array(feature_data)';
% convert tables to matrices
% target_data = target_data{:,:}';
% %target_data = str2double(target_data);
% feature_data = feature_data{:,:}';
% feature_data = str2double(feature_data);
% func = @(var) round(var,0); % set rounding fxn
% varfun(func, features(3,:)); % round target data to be binary
%% k-fold cross validation

 net = feedforwardnet(10,'trainscg');
 preds_l2out = leave_two_out(net, feature_data, target_data);


 %% QDA testing
 
predictions = struct('val_targets',{}, 'val_preds',{}, 'train_targets',{}, 'train_preds',{});
features = feature_data(1:end,2:end);
model_num = 0;
for ii = 1:5
    train_inds = find(features(1,:) ~= ii);
    val_inds = find(features(1,:) == ii);
    train_features = features(2:end, train_inds)';
    val_features = features(2:end, val_inds)';
    train_targets = target_data(train_inds);
    val_targets = target_data(val_inds);

    train_preds = classify(train_features, train_features, train_targets, 'quadratic');
    val_preds = classify(val_features, train_features, train_targets, 'quadratic');
    
    %predictions = [];
    model_num = model_num + 1;
    disp(model_num)
    predictions(model_num).val_targets = val_targets;
    predictions(model_num).val_preds = val_preds';
    predictions(model_num).train_targets = train_targets;
    predictions(model_num).train_preds = train_preds';
    
end





