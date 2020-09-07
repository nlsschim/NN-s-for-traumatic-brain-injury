function predictions = leave_two_out(net, feature_data, target_data)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
predictions = struct('val_targets',{}, 'val_preds',{}, 'train_targets',{}, 'train_preds',{});
features = feature_data(1:end,2:end);
model_num = 0;

% 
% I think I know why this loop is only [1,29] when the others are [1,30] -- because the leave-2-out
%  needs two consecutive iterations, of which there are only 29 -- but it would still be nice to have
%  a comment about it.
% 
% Also, you may want 'main.m' to define a global variable 'ITERATION_COUNT=30' that each of your functions
%  takes in as a parameter (and then this one in particular would use 'ITERATION_COUNT - 1'). This will
%  let you easily modify that number globally if/when you want to, and eliminate these 'magic' numbers
%  that have exact values (i.e. '29') without an exact description (i.e. 'ITERATION_COUNT'). 
% 

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
    train_features = features(2:end, train_inds);
    val_features = features(2:end, val_inds);
    train_targets = target_data(train_inds);
    val_targets = target_data(val_inds);

    
    
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

end
