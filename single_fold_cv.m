function [predictions] = single_fold_cv(net, feature_data, target_data)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
predictions = struct('val_targets',{}, 'val_preds',{}, 'train_targets',{}, 'train_preds',{});
features = feature_data(2:end,2:end);
for ii=1:30
    
    disp(ii)
    
    train_inds = find(features(1,:) ~= ii);
    val_inds = find(features(1,:) == ii);
    train_features = features(2:end, train_inds);
    val_features = features(2:end, val_inds);
    train_targets = target_data(train_inds);
    val_targets = target_data(val_inds);
    
    %net = patternnet(10);
    [net,tr] = train(net, train_features, train_targets);
    train_preds = net(train_features);
    val_preds = net(val_features);
    
    %predictions = [];
    predictions(ii).val_targets = val_targets;
    predictions(ii).val_preds = val_preds;
    predictions(ii).train_targets = train_targets;
    predictions(ii).train_preds = train_preds;
    

end
end

