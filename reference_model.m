clear all
% Read image and show it on figure
A = imread("coins.png");
imshow(A)

% Define range of Rs we're looking for
rmin = 20;
rmax = 30;

% Execute the function to finding circles
[centers, rads] = imfindcircles(A, [rmin rmax]);

% Draw found circles
hold on;
viscircles(centers, rads, 'EdgeColor', 'r');

%% Reference model:

B = imread("coins.png");
I = edge(B); % no need to conv to gray because already is
threshhold = 7;
Rtable = 20:1:30;
output = zeros(length(Rtable), size(I, 1), size(I, 2));

for R = Rtable
    R_with_thresh = R + threshhold;
    r = R_with_thresh;
    if r < threshhold + 2
        r = 0;
    else
        r = r - threshhold - 2;
    end
    dim = 2 * R_with_thresh + 1;

    
    buffer = false(dim, dim);
    for i = 1:dim
        for j = 1:dim
            distance = (i - ceil(dim / 2))^2 + (j - ceil(dim / 2))^2;
            if distance <= R_with_thresh^2 && distance > r^2
                buffer(i, j) = true;
            end
        end
    end
    if r == 0 && R_with_thresh > r
        buffer(ceil(dim / 2), ceil(dim / 2)) = true;
    end

    output_temp = zeros(size(I, 1) + 2*R_with_thresh, size(I, 2) + 2*R_with_thresh);
    for w = 1:size(I, 1)
        for h = 1:size(I, 2)
            if I(w, h) == 1
                output_temp(w:w+2*R_with_thresh, h:h+2*R_with_thresh) = ...
                    output_temp(w:w+2*R_with_thresh, h:h+2*R_with_thresh) + buffer;
            end
        end
    end

    idx = R - Rtable(1) + 1;
    temp = output_temp(R_with_thresh+1:end-R_with_thresh, R_with_thresh+1:end-R_with_thresh);
    assert(all(size(temp) == [size(I, 1), size(I, 2)]), 'Dimension mismatch!');
    output(idx, :, :) = reshape(temp, 1, size(I, 1), size(I, 2));
    output(idx, :, :) = output(idx, :, :) ./ max(squeeze(output(idx, :, :)), [], 'all');
end

%%
figure; 
for idx = 1:length(Rtable)
    subplot(ceil(sqrt(length(Rtable))), ceil(sqrt(length(Rtable))), idx);
    imshow(mat2gray(squeeze(output(idx, :, :))));
    R = Rtable(idx);
    title(sprintf('R = %d', R));
end
sgtitle("HoughSpace")

%% 
figure;
distanceThreshold = 60; 

centers = [];
radii = [];
hs = [];

for idx = 1:length(Rtable)
    houghSpace = squeeze(output(idx, :, :));
    threshold = 0.5 * max(houghSpace(:));
    [rows, cols] = find(houghSpace > threshold);
    centers = [centers; cols, rows];
    radii = [radii; repmat(Rtable(idx), length(rows), 1)];
    
    for k = 1:length(rows)
        hs = [hs; houghSpace(rows(k), cols(k))];
    end
end

[~, sortIdx] = sort(hs, 'descend');
sortedCenters = centers(sortIdx, :);
sortedRadii = radii(sortIdx);
sortedHs = hs(sortIdx);

filteredCenters = [];
filteredRadii = [];
filteredHs = [];
used = false(size(sortedCenters, 1), 1); % Flaga użycia centrum

for i = 1:size(sortedCenters, 1)
    if ~used(i)
        distances = sqrt(sum((sortedCenters - sortedCenters(i, :)).^2, 2));
        filteredCenters = [filteredCenters; sortedCenters(i, :)];
        groupIdx = find(distances < distanceThreshold);
        averageRadius = mean(sortedRadii(groupIdx));
        averageHs = mean(sortedHs(groupIdx));
        
        filteredRadii = [filteredRadii; sortedRadii(i)*0.5 + averageRadius*0.5];
        filteredHs = [filteredHs; sortedHs(i)];



        used(distances < distanceThreshold) = true;
    end
end

imshow(B);
hold on;
viscircles(filteredCenters, filteredRadii, 'EdgeColor', 'r');
title('Okręgi z największymi wartościami w przestrzeni Hougha dla podobnych centrów');

