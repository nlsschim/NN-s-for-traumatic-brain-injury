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
features = feature_data{:,:}';
% func = @(var) round(var,0); % set rounding fxn
% varfun(func, features(3,:)); % round target data to be binary

%% split into train and validation
first_29 = features(features(2,:) ~= string(30));
S=size(first_29);
len = S(2);
train_features = features(3:12, 2:len);
train_features = str2double(train_features);
train_targets = target_data(:,2:len);

validation_features = features(3:12, len+1:end);
validation_features = str2double(validation_features);
validation_targets = target_data(:,len+1:end);

net = patternnet(10);
[net,tr] = train(net, train_features, train_targets);
plotperform(tr)

%% cross validation

validation_preds = net(validation_features);
testClasses = validation_preds > 0.2;
plotconfusion(validation_targets, testClasses)
%plotroc(validation_targets, validation_preds)