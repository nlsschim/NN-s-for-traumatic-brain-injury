function [feature_data, target_data] = tpi_csv_preprocess(filepath)

feature_data = import_features(filepath); % extract features from csv
target_data = import_target(filepath); % extract targets from csv

func = @(var) round(var,0); % set rounding fxn
target_data = varfun(func, target_data); % round target data to be binary

% put data in table format
writetable(target_data, 'target_data.csv', 'Delimiter',',','QuoteStrings',true);
writetable(feature_data, 'feature_data.csv', 'Delimiter', ',', 'QuoteStrings', true);

% get data to be doubles
target_data = table2array(target_data)';
feature_data = table2array(feature_data)';
% convert tables to matrices
% target_data = target_data{:,:}';
% %target_data = str2double(target_data);
% feature_data = feature_data{:,:}';
% feature_data = str2double(feature_data);
% func = @(var) round(var,0); % set rounding fxn
% varfun(func, features(3,:)); % round target data to be binary


end





