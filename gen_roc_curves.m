function [X,Y,T,AUC] = gen_roc_curves(predictions)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

posClass = 1;
figure
hold on
%subplot(2,1,1)
for ii=1:30
    
    train_labels = predictions(ii).train_targets;
    train_scores = predictions(ii).train_preds;
    %val_labels = predictions(ii).val_targets;
    %val_scores = predictions(ii).val_preds;
    
    [X,Y,T,AUC] = perfcurve(train_labels, train_scores, posClass);
    %[X1,Y1] = perfcurve(val_labels, val_scores, posClass);

    %subplot(2,1,1);
    plot(X, Y)
    %plot([0,1], [0,1], 'b', 'Linewidth',2)
    title('Training ROC curve');
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');

    % subplot(2,1,2);
    % plot(X1, Y1)
    % %plot([0,1], [0,1], 'b', 'Linewidth',2)
    % title('Validation ROC curve');
    % xlabel('False Positive Rate');
    % ylabel('True Positive Rate');

end
hold off

%subplot(2,1,2);
figure
hold on
for ii=1:30
    val_labels = predictions(ii).val_targets;
    val_scores = predictions(ii).val_preds;
    [X1,Y1] = perfcurve(val_labels, val_scores, posClass);

    
    plot(X1, Y1)
    %plot([0,1], [0,1], 'b', 'Linewidth',2)
    title('Validation ROC curve');
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');

end

hold off


end

