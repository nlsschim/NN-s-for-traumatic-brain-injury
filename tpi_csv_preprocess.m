%% preprocess and load data

% Specify filepath for csv data
filepath = '/Users/nlsschim/Documents/LAB/ML_TBI/john_data/DoD012_TPIHarmonicsTable.csv';
feature_data = import_features(filepath); % extract features from csv
target_data = import_target(filepath); % extract targets from csv

func = @(var) round(var,0); % set rounding fxn
target_data = varfun(func, target_data); % round target data to be binary

writetable(target_data, 'target_data.csv', 'Delimiter',',','QuoteStrings',true);
writetable(feature_data, 'feature_data.csv', 'Delimiter', ',', 'QuoteStrings', true);

% convert tables to matrices
target_data = target_data{:,:}';
feature_data = feature_data{:,:}';
% func = @(var) round(var,0); % set rounding fxn
% varfun(func, features(3,:)); % round target data to be binary
%% k-fold cross validation


%% split into train and validation

%figure 
%hold on
%index_count = 2;
% predictions = struct('val_targets',{}, 'val_preds',{}, 'train_targets',{}, 'train_preds',{});
% features = feature_data(2:end,2:end);
% for ii=1:30
%     
%     disp(ii)
%     
%     train_inds = find(features(1,:) ~= string(ii));
%     val_inds = find(features(1,:) == string(ii));
%     train_features = str2double(features(2:end, train_inds));
%     val_features = str2double(features(2:end, val_inds));
%     train_targets = target_data(train_inds);
%     val_targets = target_data(val_inds);
%     
%     net = patternnet(10);
%     [net,tr] = train(net, train_features, train_targets);
%     train_preds = net(train_features);
%     val_preds = net(val_features);
%     
%     %predictions = [];
%     predictions(ii).val_targets = val_targets;
%     predictions(ii).val_preds = val_preds;
%     predictions(ii).train_targets = train_targets;
%     predictions(ii).train_preds = train_preds;
%     
% 
% end

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

predictions = neural_net(netOne, feature_data, target_data);
%% more preds
netTwo = patternnet(8);
preds_eight = neural_net(netTwo, feature_data, target_data);

%% preds

%[x, Y] = gen_roc_curves(predictions, 4);
[X, Y] = gen_roc_curves(preds_eight, 8);

%% different learning rates

netFive = patternnet(10);
netFive.trainParam.epochs = 50;
preds_netFive = neural_net(netFive, feature_data, target_data);
% 
%% 
[x, y] = gen_roc_curves(preds_netFive, 10);
%% another learning rate change
netSix = patternnet(10);
netSix.trainParam.min_grad = 0.01;
preds_netSix = neural_net(netSix, feature_data, target_data);
[x, y] = gen_roc_curves(preds_netSix, 10);
%% basic normal net
basic_net = patternnet(10);
preds_basic_net = neural_net(basic_net, feature_data, target_data);
[x, y] = gen_roc_curves(preds_basic_net, 10);

