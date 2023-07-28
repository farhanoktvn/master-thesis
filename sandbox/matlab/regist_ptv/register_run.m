function register_run(sample_id, run_id, imgs, reg_dir)
    % Init save directory
    save_dir = reg_dir + "/" + sample_id + "/" + run_id;
    mkdir(save_dir);

    % Import images
    source_imgs = zeros(1024, 1280, length(imgs));
    for i = 1:length(imgs)
        wavelength = 430 + i*10;
        filename = imgs(i).folder + "/" + imgs(i).name;
        img = imread(filename);
        img = im2gray(img);
        source_imgs(:, :, i) = img;
    end
    source_imgs = double(reshape(source_imgs, [size(source_imgs, 1), size(source_imgs, 2), 1, 1, size(source_imgs,3)]));
%     source_imgs = sqrt(double(reshape(source_imgs, [size(source_imgs, 1), size(source_imgs, 2), 1, 1, size(source_imgs,3)])));
%     %sqrt is a dirty trick to get rid of highlights
    source_imgs = source_imgs ./ max(max(source_imgs, [], 1), [], 2);


    % Register stack
    opts = [];
    opts.pix_resolution = [5, 5];
    opts.metric = 'nuclear';
    
    opts.grid_spacing = [10, 10];
    opts.isoTV = 5e-3;
    
    opts.spline_order = 1;
    opts.max_iters = 70;
    
    opts.display = 'off';
    
    opts.border_mask = 6;
    opts.k_down = 0.6;
    
    
    tic
    [voldef_pl, Tmin_pl,  Kmin_pl] = ptv_register(source_imgs, [], opts);
    toc


    % Save registered images
    voldef_pl = squeeze(voldef_pl);
    for i=1:size(voldef_pl, 3)
        imwrite(voldef_pl(:,:,i), save_dir + "/" + imgs(i).name);
    end
end

