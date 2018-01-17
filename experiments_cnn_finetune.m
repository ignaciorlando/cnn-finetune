
exp_layers_for_update = { ...
    [14, 16, 18] ... % fully connected layers + classifier
};

numEpochs = 120;
learningRate = 0.00001;
dropout = 0.5;
weight_decay = 1e-4;

for i = 1 : length(exp_layers_for_update)
    [net, info] = cnn_finetune('origa650', 'layers_for_update', exp_layers_for_update{i}, ...
                                           'numEpochs', numEpochs, ...
                                           'learningRate', learningRate, ...
                                           'dropout', dropout, ...
                                           'weightDecay', weight_decay);
end