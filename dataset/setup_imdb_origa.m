function [imdb] = setup_imdb_origa( dataset_folder, image_source,  varargin )
%
    opts.seed = 1 ;
    opts.datasetName = '';
    opts = vl_argparse(opts, varargin);

    % open filenames and labels
    load(fullfile(dataset_folder, 'labels.mat'));
    
    labels = double(labels) + 1;
    
    % number of the data sets
    imdb.sets = {'train', 'val'};
    % setup image folder
    imdb.imageDir = fullfile(dataset_folder, image_source);
    % classes ids
    imdb.classes.name = unique(labels);
    imdb.classes.description = { 'non-glaucomatous', 'glaucomatous' };
    % images ids
    imdb.images.name = filenames;
    imdb.images.label = labels;
    imdb.images.id = 1:1:length(labels);
    imdb.images.set = ones(length(labels), 1);
    % setup the validation set
    for i = 1 : length(imdb.classes.name)
        % get the amount of samples with this label
        n_samples = length(find(labels==imdb.classes.name(i)));
        % get the 10% for validation
        n_val_samples = floor(0.1 * n_samples);
        % sample n_val_samples for the ids with the given label
        val_ids = datasample(find(labels==imdb.classes.name(i)), n_val_samples, 'Replace', false);
        % these samples will be labeled as part of the validation set
        imdb.images.set(val_ids) = 2;
    end
    % complete with metadata
    imdb.meta.classes = imdb.classes.name;
    imdb.meta.inUse = true(size(imdb.images.name));

end

