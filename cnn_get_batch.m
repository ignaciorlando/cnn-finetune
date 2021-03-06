function imo = cnn_get_batch(images, varargin)
% Modified from CNN_IMAGENET_GET_BATCH
%
% - added `pad` option
% - deals with images of types other than jpeg
% -------------------------------------------------------
% Re-Modified by Armin khodaei (Arminkhodaei@gmail.com)
%
% - flexible image channels count is added
% - 'dataType' option can be choosed between 'cell' and 'array'
%

    opts.imageSize = [28, 28, 1] ; % [HEIGHT WIDTH CHANNEL]
    opts.border = [29, 29] ;
    opts.pad = 0;  % [TOP BOTTOM LEFT RIGHT]
    opts.keepAspect = true ;
    opts.numAugments = 1 ;
    opts.transformation = 'none' ;
    opts.averageImage = [] ;
    opts.rgbVariance = zeros(0,3,'single') ;
    opts.interpolation = 'bilinear' ;
    opts.numThreads = 1 ;
    opts.prefetch = false ;
    opts.numImages = 0;
    opts.dataType = 'cell';
    opts = vl_argparse(opts, varargin);

    % if only one value is given, apply the same amount of padding to all borders
    if numel(opts.pad)==1, opts.pad = repmat(opts.pad,[1 4]); end
    if numel(opts.border)==1, opts.border = repmat(opts.border,[1 2]); end

    isJpg = 0;
    fetch = 0;
    if numel(size(images)) == 4, opts.numImages = size(images, 4); opts.dataType = 'array'; end
    if numel(size(images)) == 2 && iscell(images)
        opts.numImages = numel(images);

        % fetch is true if images is a list of filenames (instead of
        % a cell array of images Or a 4-D array of images)
        fetch = opts.numImages >= 1 && ischar(images{1});

        % isJpg is true if all images to fetch are of jpeg format
        isJpg = fetch && strcmpi(images{1}(end-3:end),'.jpg');
    end

    % choose how to load images data
    if isJpg
        if opts.prefetch
            vl_imreadjpeg(images, 'numThreads', opts.numThreads, 'prefetch');
            imo = [];
            return;
        end
        if fetch
            im = vl_imreadjpeg(images,'numThreads', opts.numThreads);
        end
    else
        im = images;
    end

    tfs = [] ;
    switch opts.transformation
        case 'none'
            tfs = [
                .5 ;
                .5 ;
                0 ] ;
        case 'f5'
            tfs = [...
                .5 0 0 1 1 .5 0 0 1 1 ;
                .5 0 1 0 1 .5 0 1 0 1 ;
                0 0 0 0 0  1 1 1 1 1] ;
        case 'f25'
            [tx,ty] = meshgrid(linspace(0,1,5)) ;
            tfs = [tx(:)' ; ty(:)' ; zeros(1,numel(tx))] ;
            tfs_ = tfs ;
            tfs_(3,:) = 1 ;
            tfs = [tfs,tfs_] ;
        case 'stretch'
        otherwise
            error('Uknown transformations %s', opts.transformation) ;
    end
    [~,transformations] = sort(rand(size(tfs,2), opts.numImages), 1) ;

    if ~isempty(opts.rgbVariance) && isempty(opts.averageImage)
        opts.averageImage = zeros(1,1,3) ;
    end
    if numel(opts.averageImage) == 3
        opts.averageImage = reshape(opts.averageImage, 1,1,3) ;
    end

    imo = zeros(opts.imageSize(1), opts.imageSize(2), opts.imageSize(3), ...
        opts.numImages * opts.numAugments, 'single');

    si = 1 ;
    for i=1:opts.numImages

        % acquire image
        switch opts.dataType
            case 'cell'
                imt = imread(im{i}) ;
                imt = single(imt) ; % faster than im2single (and multiplies by 255)
                if size(imt,3) == 1
                    imt = cat(3, imt, imt, imt) ;
                end

            case 'array'
                imt = im(:, :, :, i);
                imt = single(imt) ; % faster than im2single (and multiplies by 255)
                if size(imt,3) == 1
                    imt = cat(3, imt, imt, imt) ;
                end
        end

        % pad
        if ~isempty(opts.pad) && any(opts.pad>0)
            imtt = imt;
            imt = 255*ones(size(imtt,1)+sum(opts.pad(1:2)), ...
                size(imtt,2)+sum(opts.pad(3:4)), 3, 'like', imtt);
            imt(opts.pad(1)+(1:size(imtt,1)), opts.pad(3)+(1:size(imtt,2)),:) = imtt;
        end

        % resize
        w = size(imt,2) ;
        h = size(imt,1) ;
        factor = [(opts.imageSize(1)+opts.border(1))/h ...
            (opts.imageSize(2)+opts.border(2))/w];

        if opts.keepAspect
            factor = max(factor) ;
        end
        if any(abs(factor - 1) > 0.0001)
            imt = imresize(imt, factor, opts.interpolation) ;
        end

        % crop & flip
        w = size(imt,2) ;
        h = size(imt,1) ;
        for ai = 1:opts.numAugments
            switch opts.transformation
                case 'stretch'
                    sz = round(min(opts.imageSize(1:2)' .* (1-0.1+0.2*rand(2,1)), [h;w])) ;
                    dx = randi(w - sz(2) + 1, 1) ;
                    dy = randi(h - sz(1) + 1, 1) ;
                    flip = rand > 0.5 ;
                otherwise
                    tf = tfs(:, transformations(mod(ai-1, numel(transformations)) + 1)) ;
                    sz = opts.imageSize(1:2) ;
                    dx = floor((w - sz(2)) * tf(2)) + 1 ;
                    dy = floor((h - sz(1)) * tf(1)) + 1 ;
                    flip = tf(3) ;
            end
            sx = round(linspace(dx, sz(2)+dx-1, opts.imageSize(2))) ;
            sy = round(linspace(dy, sz(1)+dy-1, opts.imageSize(1))) ;
            if flip
                sx = fliplr(sx);
            end

            if ~isempty(opts.averageImage)
                offset = opts.averageImage ;
                if ~isempty(opts.rgbVariance)
                    offset = bsxfun(@plus, offset, reshape(opts.rgbVariance * randn(3,1), 1,1,3)) ;
                end
                imo(:,:,:,si) = bsxfun(@minus, imt(sy,sx,:), offset) ;
            else
                imo(:,:,:,si) = imt(sy,sx,:) ;
            end
            si = si + 1 ;
        end
    end
    
end