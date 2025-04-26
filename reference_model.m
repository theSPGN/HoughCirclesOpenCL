% Read image and show it on figure
A = imread("coins.png");
imshow(A)

% Define range of Rs we're looking for
rmin = 20;
rmax = 30;

% Execute the function to finding circles
[centers, rads] = imfindcircles(A, [rmin rmax]);

% Draw found circles
for i = 1:length(centers)
    h = drawcircle('Center', [centers(i, 1), centers(i, 2)], 'Radius', rads(i), 'Color', 'r');
end

imwrite(A, "coins.png")