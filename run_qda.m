function predictions = run_qda(feature_data, target_data)
% This function runs QDA on the passed target and feature csv data
% using leave-one-out cross validation testing
% The function returns a struct of actual and predicted values
% for training and validation sets for each model


predictions = struct('val_targets',{}, 'val_preds',{}, 'train_targets',{}, 'train_preds',{});
features = feature_data(1:end,2:end);
model_num = 0;
for ii = 1:30
    train_inds = find(features(1,:) ~= ii);
    val_inds = find(features(1,:) == ii);
    train_features = features(2:end, train_inds)';
    val_features = features(2:end, val_inds)';
    train_targets = target_data(train_inds);
    val_targets = target_data(val_inds);

    %train_preds = classify(train_features, train_features, train_targets, 'quadratic');
    %val_preds = classify(val_features, train_features, train_targets, 'quadratic');
    
    Mdl = fitcdiscr(train_features, train_targets, 'DiscrimType', 'quadratic');
    [~,train_preds,~] = predict(Mdl,train_features);
    [~,val_preds,~] = predict(Mdl,val_features);
    
    %predictions = [];
    model_num = model_num + 1;
    disp(model_num) % used to show how many models are left to train
    predictions(model_num).val_targets = val_targets;
    predictions(model_num).val_preds = val_preds(:,1); 
    predictions(model_num).train_targets = train_targets;
    predictions(model_num).train_preds = train_preds(:,1);
    
end
end

