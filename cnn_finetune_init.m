function net = cnn_finetune_init(imdb, net)

    % setup the initialization method
    opts.weightInitMethod = 'xavierimproved' ;
    opts.scale = 1; 

    % by default, we use vgg-m
    if ~exist('net', 'var') || isempty(net)
      net = 'imagenet-matconvnet-vgg-m';
    end

    % if the network is a string...
    if ischar(net)
        % download the model first
        net_path = fullfile('data','models',[net '.mat']);
        if ~exist(net_path,'file')
            fprintf('Downloading model (%s) ...', net) ;
            vl_xmkdir(fullfile('data','models')) ;
            urlwrite(fullfile('http://www.vlfeat.org/matconvnet/models', [net '.mat']), net_path) ;
            fprintf(' done!\n');
        end
        % load the cnn
        net = load(net_path);
    end
    
    % replace the last layer by a loss function
    net.layers{end} = struct('name','loss','type','softmaxloss'); 

    % copy the original parameters to pretrain
    if ~isfield(net.meta, 'pretrain')
        net.meta.pretrain = {};
    end
    net.meta.pretrain = [net.meta.pretrain net.meta];
    net.meta.pretrain{end} = rmfield(net.meta.pretrain{end}, 'pretrain'); 
    net.meta.classes.name = imdb.meta.classes;
    net.meta.classes.description = imdb.meta.classes;

    
    fc_layers = find(cellfun(@(s) numel(s.name)>=2 && strcmp(s.name(1:2),'fc'),net.layers));
    last_fc_layer = max(fc_layers);
    n_fc_units = 2048;
    
    for i = 1 : length(fc_layers)
        % get the size of the last fully connected layer
        [h, w, in , out] = size(net.layers{fc_layers(i)}.weights{1});
        % replace the last fully connected layer by the appropriate one
        if fc_layers(i) == last_fc_layer
            in = n_fc_units;
            out = numel(net.meta.classes.name); 
        else
            if h == 1
                in = n_fc_units;
            end
            out = n_fc_units;
        end
        net.layers{fc_layers(i)}.weights = {init_weight(opts, h, w, in, out, 'single'), zeros(out, 1, 'single')};
    end

end




% -------------------------------------------------------------------------
function weights = init_weight(opts, h, w, in, out, type)
% -------------------------------------------------------------------------
% See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
% rectifiers: Surpassing human-level performance on imagenet
% classification. CoRR, (arXiv:1502.01852v1), 2015.

    switch lower(opts.weightInitMethod)
      case 'gaussian'
        sc = 0.01/opts.scale ;
        weights = randn(h, w, in, out, type)*sc;
      case 'xavier'
        sc = sqrt(3/(h*w*in)) ;
        weights = (rand(h, w, in, out, type)*2 - 1)*sc ;
      case 'xavierimproved'
        sc = sqrt(2/(h*w*out)) ;
        weights = randn(h, w, in, out, type)*sc ;
      otherwise
        error('Unknown weight initialization method''%s''', opts.weightInitMethod) ;
    end

end
