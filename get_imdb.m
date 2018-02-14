function [ imdb ] = get_imdb( datasetName, image_source, varargin )
%GET_IMDB Get imdb structure for the specified dataset


args.func = @setup_imdb_origa;
args.rebuild = false;
args = vl_argparse(args,varargin);

imdb_filename = strcat('imdb-',image_source,'.mat');

datasetDir = fullfile('data',datasetName);
imdbPath = fullfile(datasetDir, imdb_filename);

if ~exist(datasetDir,'dir'), 
    error('Unknown dataset: %s', datasetName);
end

if exist(imdbPath,'file') && ~args.rebuild, 
    imdb = load(imdbPath);
else
    imdb = args.func(datasetDir, image_source);
    save(imdbPath,'-struct','imdb');
end

end

