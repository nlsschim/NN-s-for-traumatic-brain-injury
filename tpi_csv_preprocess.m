%% preprocess and load data

% Specify filepath for csv data
filepath = '/Users/nlsschim/Documents/LAB/ML_TBI/DoD012_TPIHarmonicsTable (1).csv';
feature_data = import_features(filepath); % extract features from csv
target_data = import_target(filepath); % extract targets from csv

func = @(var) round(var,0); % set rounding fxn
target_data = varfun(func, target_data); % round target data to be binary

writetable(target_data, 'target_data.csv', 'Delimiter',',','QuoteStrings',true);
writetable(feature_data, 'feature_data.csv', 'Delimiter', ',', 'QuoteStrings', true);

% convert tables to matrices
target_data = target_data{:,:}';
%target_data = str2double(target_data);
feature_data = feature_data{:,:}';
feature_data = str2double(feature_data);
% func = @(var) round(var,0); % set rounding fxn
% varfun(func, features(3,:)); % round target data to be binary
%% k-fold cross validation

 net = feedforwardnet(10,'trainscg');
 preds_l2out = leave_two_out(net, feature_data, target_data);
%% split into train and validation

%figure 
%hold on
%index_count = 2;
predictions = struct('val_targets',{}, 'val_preds',{}, 'train_targets',{}, 'train_preds',{});
features = feature_data(1:end,2:end);
model_num = 0;
for ii=1:29
    
    
%     first_train_inds = find(features(1,:) ~= string(ii));
%     first_val_inds = find(features(1,:) == string(ii));
%     train_features = str2double(features(2:end, first_train_inds));
%     first_val_features = str2double(features(2:end, first_val_inds));
%     first_train_targets = target_data(first_train_inds);
%     first_val_targets = target_data(first_val_inds);
    
    for jj=(ii+1):30
    
    
    train_inds = find(features(1,:) ~= jj | ii);
    val_inds = find(features(1,:) == jj | ii);
    train_features = features(3:end, train_inds);
    val_features = features(3:end, val_inds);
    train_targets = target_data(train_inds);
    val_targets = target_data(val_inds);

    
    net = feedforwardnet(10,'trainscg');
    net.trainParam.showWindow = 1;
    %net.trainParam.min_grad = 0.001;
    net.trainParam.epochs = 100;
    [net,tr] = train(net, train_features, train_targets);
    train_preds = net(train_features);
    val_preds = net(val_features);
    
    %predictions = [];
    model_num = model_num + 1;
    disp(model_num)
    predictions(model_num).val_targets = val_targets;
    predictions(model_num).val_preds = val_preds;
    predictions(model_num).train_targets = train_targets;
    predictions(model_num).train_preds = train_preds;
    
    end
end

%% cross validation


%% trying stuff out
% figure
% hold on
% title('Training ROC curve');
% xlabel('False Positive Rate');
% ylabel('True Positive Rate');
% for ii=1:30
% labels = predictions(ii).train_targets;
% scores = predictions(ii).train_preds;
% posClass = 1;
% [X,Y] = perfcurve(labels, scores, posClass);
% plot(X,Y)
% end
% hold off
% 
% figure
% hold on
% title('Validation ROC curve');
% xlabel('False Positive Rate');
% ylabel('True Positive Rate');
% for ii=1:30
% labels = predictions(ii).val_targets;
% scores = predictions(ii).val_preds;
% posClass = 1;
% [X,Y] = perfcurve(labels, scores, posClass);
% plot(X,Y)
% end
% hold off

%% bunch of nets
netOne = patternnet(4);
% netTwo = patternnet(8);
% netThree = patternnet(12);
% netFour = patternnet(16);

predictions = single_fold_cv(netOne, feature_data, target_data);
%% more preds
netTwo = patternnet(8);
preds_eight = single_fold_cv(netTwo, feature_data, target_data);

%% preds

%[x, Y] = gen_roc_curves(predictions, 4);
[X, Y] = gen_roc_curves(preds_eight, 8);

%% different learning rates

netFive = patternnet(10);
netFive.trainParam.epochs = 50;
preds_netFive = single_fold_cv(netFive, feature_data, target_data);
% 
%% 
[x, y] = gen_roc_curves(preds_netFive, 10);
%% another learning rate change
netSix = patternnet(10);
netSix.trainParam.min_grad = 0.01;
preds_netSix = single_fold_cv(netSix, feature_data, target_data);
[x, y] = gen_roc_curves(preds_netSix, 10);
%% basic normal net
basic_net = patternnet(10);
preds_basic_net = single_fold_cv(basic_net, feature_data, target_data);
[x, y] = gen_roc_curves(preds_basic_net, 10);

