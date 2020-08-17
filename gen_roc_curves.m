function [X,Y] = gen_roc_curves(predictions, number_neurons)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
hidden_neurons = number_neurons;
figure
hold on
for ii=1:30
labels = predictions(ii).train_targets;
scores = predictions(ii).train_preds;
posClass = 1;
[X,Y] = perfcurve(labels, scores, posClass);
plot(X, Y)
plot([0,1], [0,1], 'b', 'Linewidth',2)
title('Training ROC curve: Hidden Neurons = ' + string(hidden_neurons));
xlabel('False Positive Rate');
ylabel('True Positive Rate');
end
hold off 

figure
hold on
for ii=1:30
labels = predictions(ii).val_targets;
scores = predictions(ii).val_preds;
posClass = 1;
[X1,Y1] = perfcurve(labels, scores, posClass);
plot(X1, Y1)
plot([0,1], [0,1], 'b', 'Linewidth',2)
title('Validation ROC curve: Hidden Neurons = ' + string(hidden_neurons));
xlabel('False Positive Rate');
ylabel('True Positive Rate');
end
hold off
end

