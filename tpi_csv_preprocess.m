
% Specify filepath for csv data
filepath = '/Users/nlsschim/Documents/LAB/ML_TBI/June_data/DoD001_TPIHarmonicsTable.csv';
feature_data = import_features(filepath); % extract features from csv
target_data = import_target(filepath); % extract targets from csv

func = @(var) round(var,0); % set rounding fxn
target_data = varfun(func, target_data); % round target data to be binary

writetable(target_data, 'target_data.csv', 'Delimiter',',','QuoteStrings',true)
writetable(feature_data, 'feature_data.csv', 'Delimiter', ',', 'QuoteStrings', true)


