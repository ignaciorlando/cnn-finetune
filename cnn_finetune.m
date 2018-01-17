function [net, info] = cnn_finetune(datasetName, varargin)

    % default parameters
    opts.expDir = fullfile('data','exp') ;
    opts.baseNet = 'imagenet-matconvnet-vgg-m';
    opts.numEpochs = 40; 
    opts.numFetchThreads = 12 ;
    opts.imdb = [];
    opts.includeVal = false; 
    opts.aug = 'stretch'; 
    opts.border = 0; 
    opts.pad = 0; 
    opts.datafn = @setup_imdb_origa; % the data function will load files from origa
    opts.layers_for_update = [ 19 ];
    opts.learningRate = 0.00001;
    opts.dropout = 0.5;
    opts.weightDecay = 0.0001;
    [opts,varargin] = vl_argparse(opts, varargin) ;

    opts.train.dropout = opts.dropout;
    opts.train.momentum = 0.9;
    opts.train.batchSize = 16;
    opts.train.maxIterPerEpoch = Inf; 
    opts.train.balancingFunction = {[], []};
    opts.train.gpus = [];
    opts.train = vl_argparse(opts.train, varargin) ;

    if ~exist(opts.expDir, 'dir')
        vl_xmkdir(opts.expDir)
    end

    % ---------------------------------------------------
    % Prepare data & model
    % ---------------------------------------------------

    % load the database
    if isempty(opts.imdb) 
      imdb = get_imdb(datasetName); 
    else
      imdb = opts.imdb;
    end

    % identify training and validation sets
    opts.train.train = find(imdb.images.set==1);
    opts.train.val = find(imdb.images.set==2); 
    if opts.includeVal
      opts.train.train = [opts.train.train opts.train.val];
      opts.train.val = [];
    end

    % initialize the CNN from the base network
    net = cnn_finetune_init(imdb, opts.baseNet); 
    if opts.dropout > 0
        net = addDropout(net, opts);
    end

    % ---------------------------------------------------
    % Learn
    % ---------------------------------------------------

    % identify trainable layers
    trainable_layers = find(cellfun(@(l) isfield(l,'learningRate'),net.layers)); 
    % identify fully connected layers
    fc_layers = find(cellfun(@(s) numel(s.name)>=2 && strcmp(s.name(1:2),'fc'),net.layers));
    fc_layers = intersect(fc_layers, trainable_layers);
    % get learning rates on each layer
    lr = cellfun(@(l) l.learningRate, net.layers(trainable_layers),'UniformOutput',false); 
    
    % set layers for update
    layers_for_update = opts.layers_for_update; 

    % the learning rate of the layers that must not be trained is set to zero 
    for i = 1:numel(trainable_layers)
        l = trainable_layers(i); 
        if ismember(l,layers_for_update)
            net.layers{l}.learningRate = lr{i}; 
        else
            net.layers{l}.learningRate = lr{i}*0; 
        end
    end
    
    % setup the training options
    net.meta.trainOpts.learningRate = ones(opts.numEpochs, 1) * opts.learningRate;
    
    % finetune the cnn
    [net, info] = cnn_finetune_train(net, imdb, getBatchFn(opts, net.meta), ...
                                     'expDir', opts.expDir, ...
                                     net.meta.trainOpts, ...
                                     opts.train, ...
                                     'numEpochs', opts.numEpochs) ;

    % ---------------------------------------------------
    % Deploy
    % ---------------------------------------------------
    
    net = cnn_imagenet_deploy(net) ;
    modelPath = fullfile(opts.expDir, 'net-deployed.mat')
    save(modelPath, '-struct', 'net') ;
    
end




function fn = getBatchFn(opts, meta)

    bopts.numThreads = opts.numFetchThreads ;
    bopts.pad = opts.pad ;
    bopts.border = opts.border ;
    bopts.transformation = opts.aug ;
    bopts.imageSize = meta.normalization.imageSize ;
    bopts.averageImage = meta.normalization.averageImage ;
    bopts.rgbVariance = meta.augmentation.rgbVariance ;

    fn = @(x,y) getSimpleNNBatch(bopts,x,y) ;
    
end



function [net_to_return] = addDropout(net, opts)

    fc_layers = find(cellfun(@(s) numel(s.name)>=2 && strcmp(s.name(1:2),'fc'),net.layers));

    net_to_return.layers = cell(size(net.layers));
    net_to_return.meta = net.meta;
    fc_layers = fc_layers(1:end-1);
    
    n_drop_layers = 1;
    iterator = 1;
    for i = 1 : length(net.layers)
        net_to_return.layers{iterator} = net.layers{i};
        iterator = iterator + 1;
        if ismember(i, fc_layers)
            net_to_return.layers{iterator} = struct('name', strcat('drop', num2str(n_drop_layers)), 'type', 'dropout', 'rate', opts.dropout);
            iterator = iterator + 1;
            n_drop_layers = n_drop_layers + 1;
        end
    end

end



function [im,labels] = getSimpleNNBatch(opts, imdb, batch)

    images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
    isVal = ~isempty(batch) && imdb.images.set(batch(1)) ~= 1 ;

    if ~isVal
      % training
      im = cnn_get_batch(images, opts, 'prefetch', nargout == 0) ;
    else
      % validation: disable data augmentation
      im = cnn_get_batch(images, opts, 'prefetch', nargout == 0, 'transformation', 'none') ;
    end

    if nargout > 0
      labels = imdb.images.label(batch) ;
    end
end
