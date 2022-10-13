function analyze(inputImg, segmentationPath, figures)

% Define the user thresholds to raise anomaly alarm
rangesDict = getRangesDict();
orientationThresholds = rangesDict("orientationThresholds"); % degrees
curvatureThresholds = rangesDict("curvatureThresholds"); % mm^-1
thicknessThresholds = rangesDict("thicknessThresholds"); % mm
heightThresholds = rangesDict("heightThresholds"); % mm

% Define some image properties
imgPropDict = getImgPropDict();
mmPerPx = imgPropDict("mmPerPx"); % mm per pixel approximation to calculate real life measurements
nozzleCenter = imgPropDict("nozzleCenter"); % Coordinate (y,x) of the nozzle's center in pixels

% Define the edges for the histograms
histogramsDict = getHistogramsDict();
orientationEdges = histogramsDict("orientationEdges");
curvatureEdges = histogramsDict("curvatureEdges");
thicknessEdges = histogramsDict("thicknessEdges");
heightEdges = histogramsDict("heightEdges");

% Define some parameters for measuring
measureParamsDict = getMeasureParamsDict();
dRes = measureParamsDict("dRes"); % The step resolution to calculate angles
splineStep = measureParamsDict("splineStep"); % The step resolution to calculate splines from interstitial lines. It is the ratio between the spline length and the image width
lineStep = measureParamsDict("lineStep"); % The line segment lenght used to analyze orientation. It is the ratio between the spline length and the image width
textureWindowWidthRatio = measureParamsDict("textureWindowWidthRatio"); % The ratio used to determine the texture windows' width. It is the ratio between the window width and the image width 

interstitialR = measureParamsDict("interstitialR"); % Used to clean the interlayer line segmentations and for plotting purposes. Recommended value <= 0.25*interstitial_segmentation_diameter

gaussRatio = measureParamsDict("gaussRatio"); % Proportion of the texture window size to define the Gauss filter size

rgbImage = imread(segmentationPath); % Read the predictions from Python
grayImage = rgbImage(:,:,1); % Be sure that the predictions are single channel
[h,w] = size(grayImage); % Get the input image dimensions
grayImage = bwareafilt(imbinarize(grayImage),[round(w/20)*(4*interstitialR - 1), h*w]); % Get rid of small noise
grayImage = imclose(grayImage,strel('disk',2*interstitialR + 1)); % Close the image to deal with small line discontinuities

skeleton = bwmorph(grayImage > 0.5,'thin', inf); % Threshold image (to have binary values) and get the morphological skeleton
img = im2double(imdilate(skeleton,strel('disk',interstitialR))); % Dilate the skeleton to have fixed thick interstitial lines

% Get ROI (based on predicted inter-layer regions)
lineBorders = filter2([-1;1], img);
roi = zeros(h,w, 'logical');
for j=1:w
    column = lineBorders(:, j);
    borderPoints = find(column ~= 0);
    pairs = [];
    k=1;
    while k <= length(borderPoints)
        if column(borderPoints(k)) == -1
            for l=k+1:length(borderPoints)
                if column(borderPoints(l)) == 1
                    pairs = [pairs; borderPoints(k), borderPoints(l)];
                    k = l;
                    break
                end
            end
        end
        k = k+1;
    end
    [n_pairs, null] = size(pairs);
    for m=1:n_pairs
        roi(pairs(m, 1):pairs(m,2), j) = 1;
    end   
end
roi = imopen(roi, strel('disk',2*interstitialR)); % Separate layers, specially at borders

SE = padarray(ones(1,round(lineStep*w)),[round(lineStep*w/2),1],0,'both'); % Create structuring element for the orientation calculation
im_filt = zeros(size(img));
theta = zeros(size(img)); % This variable will save the predicted local orientations
angleAnalysisRange = 90;
for d = -angleAnalysisRange+dRes:dRes:angleAnalysisRange
	ii = imfilter(1*skeleton,imrotate(SE,d,'crop')); % Apply thhe filter with orientation d
    indx = ii>im_filt; % Find the pixels where the filter response is bigger than the previous filter responses
	theta(indx) = d; % Save the current orientation to the pixels found in the previous line
	im_filt = max(ii,im_filt); % Update the maximum response values
end

% Clean the theta values (delete borders that may be noisy)
theta = skeleton .* theta;

% Create image for UI displaying purposes (local orientations)
thetaLines = max(min(theta, orientationEdges(end)), orientationEdges(1)); % Crop values to thresholds
SE = strel('line', 2*interstitialR, 90);

% Dilate lines with negative curvature
negativeTheta = thetaLines;
negativeTheta(find(thetaLines) > 0) = 0;
negativeTheta = -imdilate(-skeleton.*negativeTheta, SE);

% Dilate lines with positive curvature
positiveTheta = thetaLines;
positiveTheta(find(thetaLines) < 0) = 0;
positiveTheta = imdilate(skeleton.*positiveTheta, SE);

thetaPlot = negativeTheta + positiveTheta; % Create final map with thick lines

mask = imdilate(skeleton, SE);
thetaAlpha = mask;

% Create images for UI displaying purposes (interstitial lines and defects)
photo = imread(inputImg);
photoGray = rgb2gray(photo);

% tic
% Calculate layers thicknesses
realDistances = bwdist(skeleton); % Get distance function
realDistances = realDistances*mmPerPx; % Transform pixels to mm
distances1 = filter2([0;-1;1],realDistances, 'valid'); % Vertical 1D derivative
disSkeleton = filter2([1;-1;0],sign(distances1), 'valid') >= 1;  % Get skeleton of the distance function
disSkeleton = padarray(disSkeleton,[2,0],0,'both');

filteredSkeleton = disSkeleton.*roi;  % Get only lines inside layers
filteredSkeleton = bwareafilt(logical(filteredSkeleton), [interstitialR^2, h*w]);  % Remove small lines which have low likelihood of being layer centers

% Create image to display
SE = strel('line', 2*interstitialR, 90);
mask = imdilate(filteredSkeleton, SE); % Mask to ignore background
disLines = imdilate(2*filteredSkeleton.*realDistances, SE); % Multiply by 2 to get diameter instead of radius; dilate to have thick line to display
disPlot = max(min(disLines, thicknessEdges(end)), thicknessEdges(1)); % Crop values to thresholds
disAlpha = mask;

% Calculate local curvatures
fSkeleton = filter2([1;-1], skeleton) .* skeleton; % Delete joint pixels between independent lines
fSkeleton = ~imbinarize(filter2(ones(5,1), fSkeleton), 1) .* fSkeleton;
[sk_lab,NN] = bwlabel(fSkeleton); % Separate and label each interstitial line
splineStep = round(splineStep * w);
dd = zeros(size(img));
kMap = zeros(size(img), 'double'); % Local curvatures will be stored here
mask = zeros(size(img)); % For display urposes
% there are NN interface lines found in the image
for ii = 1:NN
    try
        indx=find(bwmorph(sk_lab==ii,'endpoints')); % Find pixels of the current line
        [endy,endx]=find(bwmorph(sk_lab==ii,'endpoints'));
        % Measure the distance of pixels to the extreme points of the line
        M = zeros(size(img),'logical');
        M(indx(1))=true;
        dd_ii = bwdistgeodesic(sk_lab==ii,M);
        dd = max(dd,dd_ii);
        knots = [];
        % Get points to calculate the splines
        maxlen = max(max(dd_ii));
        
        if maxlen < 3 % If the line is too short, ignore it
            s = 'Element ignored.';
            continue
        end
        
        points = 0:splineStep:maxlen;
        if ~(points(end) == maxlen)
            points = [points maxlen];  % Be sure to add to last point in the curve
        end
        for jj = points  % Save the xy coordinates of the points to use as knots
            [yy,xx] = find(dd_ii==jj);
            knots = [knots, [xx(1);yy(1)]];
        end
        [e, n] = size(knots);
        
        % At this place, the variable 'pts' contains the coordinates of equidistant 
        % nodes of an interface line. The nodes are spaced by 'step' pixels. 
        % The nodes can now be used to approximate the lines (by splines) and
        % compute the curvature on.
   
        curve = cscvn(knots); % Get the spline as a parametric function (returns a structure)
        
        tkxy = []; % Parametric variable t, curvature at t, xy coordinates at t
        for t=curve.breaks(1):0.01:curve.breaks(end)
            tkxy = [tkxy; parametricCurve(curve,t)];  % parametricCurve returns t, curvature and xy positions of the given curve at t
        end
        
        % Save curvatures in an image
        for row=tkxy'
            k = row(2);
            x = round(row(3));
            y = round(row(4));
            kMap(y,x) = k;
            mask(y,x) = 1;
        end

    catch
        s = 'Element ignored.';
    end
end

kMap = kMap/mmPerPx; % Translate pixels to mm

% Create image for UI displaying purposes (local radii)
mask = imdilate(mask, strel('line', 2*interstitialR, 90));

% Dilate lines with negative curvature
negativeR = kMap;
negativeR(find(kMap) > 0) = 0;
negativeR = -imdilate(-negativeR, strel('line', 2*interstitialR, 90));

% Dilate lines with positive curvature
positiveR = kMap;
positiveR(find(kMap) < 0) = 0;
positiveR = imdilate(positiveR, strel('line', 2*interstitialR, 90));

curLines = positiveR + negativeR;  % Combine lines with positive and negative curvature
curPlot = max(min(curLines, curvatureEdges(end)), curvatureEdges(1)); % Crop values to thresholds
curAlpha = mask;


% Layer to nozzle height
[sk_lab, NN] = bwlabel(fSkeleton); % Separate and label each interstitial line
% Define the interlayer line of interest
underNozzleColumn = sk_lab(:, nozzleCenter(2));
counter = 0;
for row=1:h
   if underNozzleColumn(row) > 0
       counter = counter + 1;
       lineNumber = underNozzleColumn(row);
   end
   if counter == 2
       break
   end
end
% Calculate the relative heights
[rows, cols] = find(sk_lab == lineNumber);
nozzleHeights = rows - nozzleCenter(1);
nozzleHeights = nozzleHeights * mmPerPx;
% Prepare the image to plot
heightPlot = zeros(size(img));
heightPlot(sub2ind(size(heightPlot),rows,cols)) = nozzleHeights;
heightAlpha = imdilate(sk_lab == lineNumber, strel('line', 2*interstitialR, 90));
heightPlot = imdilate(heightPlot, strel('line', 2*interstitialR, 90));


%%%
% Texture features
[roi_lab,MM] = bwlabel(roi);
outputPath = fullfile("texture", "images");
mkdir(outputPath);
[filepath, name, ext] = fileparts(inputImg);
idx = 1;

windowWidth = round(textureWindowWidthRatio * w);
leveledImage = double(photoGray)-imgaussfilt(double(photoGray),round(windowWidth*gaussRatio));
leveledImage = uint8(leveledImage-min(leveledImage(:)));
windowNames = [];
textCoordinates = [];
borderMap = zeros(size(img),'logical');
for ii = 1:MM
    layerMask = uint8(roi_lab == ii);
    layerSkeleton = uint8(filteredSkeleton).*layerMask;
    layerDouble = im2double(leveledImage);
    layerDouble(layerMask == 0) = NaN;
    layerMask = double(layerMask);
    [sk_lab,NN] = bwlabel(layerSkeleton); % Find and label the lines between interstitial lines
    for jj = 1:NN
        indx=find(bwmorph(sk_lab==jj,'endpoints'));
        M = zeros(size(img),'logical');
        M(indx(1))=true;
        dd_ii = bwdistgeodesic(sk_lab==jj,M);
        maxLen = max(dd_ii(:));
        [y0,x0] = find(dd_ii==0); x = x0(1); % Find first pixel in layer center line
        [yF,xF] = find(dd_ii==(maxLen)); xF = xF(1); % Find first pixel in layer center line
      
        while true
            % Create current window
            startX = x;  % Left limit of the window
            endX = x + (windowWidth - 1); % Right limit; don't surpass image width
            
            if endX > xF
                break
            end
            
            bb = regionprops(layerDouble(:, startX:endX) >= 0,'BoundingBox').BoundingBox; % Get bounding box of the layer within the startX and endX
            startY = floor(bb(2));
            endY = startY + bb(4);
            
            smallWindow = layerDouble(startY:endY, startX:endX); % Window to analyze
            smallWindowSize = size(smallWindow);
            nonNanPixels = sum(sum(smallWindow >= 0));

            if nonNanPixels > 0.50*smallWindowSize(1)*windowWidth % Ignore windows with very few texture pixels according to the expected window size
                
                % Save image to disk so it can be read by Python
                finalWindow = smallWindow;
                finalWindow(isnan(finalWindow)) = 0.0;
                windowName = name + "-" + sprintf( '%03d', idx ) + ext;
                imwrite(finalWindow, fullfile(outputPath, windowName), 'Compression', 'none');
                
                % Save image info to show the predictions in the UI
                windowNames = [windowNames; windowName];
                textCoordinates = [textCoordinates; startX+interstitialR, startY+0.5*smallWindowSize(1)];   
                % To illustrate the paper only
                %rectangle('Position',[startX,startY,endX-startX,endY-startY],'LineWidth',10, 'EdgeColor', 'g')
                %
                % Find analyzed texture borders
                windowMask = zeros(size(img), 'logical');
                windowMask(startY:endY, startX:endX) = 1;
                windowMask = windowMask &  layerMask;
                borderMap = max(borderMap, windowMask - imerode(windowMask, strel('disk', interstitialR)));
                
                idx = idx + 1;
            end
            x = x + windowWidth; % Move the beggining of the sliding window
        end
    end
end
fileID = fopen(fullfile(outputPath, "matlab_flag"),'w'); % Tell python that texture predictions are needed
fprintf(fileID,'%s', '');
fclose(fileID);

while true % Wait until texture predictions are finished by Python
    if isfile(fullfile(outputPath, "python_flag"))
        break
    end
end
textureTable = readtable(fullfile(outputPath, "outputs.csv"), 'ReadVariableNames', true, 'ReadRowNames', true); % Read texture predictions
[status, message, messageid] = rmdir(outputPath, 's');  % Delete temporal files
borderMap = uint8(255*borderMap);  % Change from boolean to uint8

%%%


%%% Display all the results
[yLim, xLim] = size(img); 
% Create color map for anomaly detection
myMapColors = [0 0 1
    0 0.50 0.50
    0 1 0.50
    0 1 0
    0.50 1 0
    0.50 0.50 0
    1 0 0];
[X,Y] = meshgrid([1:3],[1:100]);  %// mesh of indices; 3 channels, 100 levels

set(0, 'currentfigure', figures(1)); 
clf(figures(1),'reset')
set(gcf,'name',inputImg,'numbertitle','off')
if ~strcmp(figures(1).WindowState,'maximized')
    figures(1).WindowState='maximized';
end

h = subplot(2,3,5);
% Create colormap to highlight defects (blue, red) and normality (green)
keyPoints = [heightEdges(1) heightThresholds(1) mean(heightThresholds) heightThresholds(end) heightEdges(end)];
keyPoints = 1 + ceil(99 * (keyPoints - keyPoints(1)) / (keyPoints(end)-keyPoints(1)));
keyPoints = [keyPoints(1) keyPoints(2)-1 keyPoints(2:4) keyPoints(4)+1 keyPoints(5)];
myMap = interp2(X(keyPoints,:),Y(keyPoints,:),myMapColors,X,Y); %// interpolate colormap
% Plot
imagesc(heightPlot,'AlphaData',heightAlpha); caxis([heightEdges(1) heightEdges(end)]); colormap(gca, myMap); axis equal; ylim([1 yLim]);set(gca,'color','black');
title("Nozzle height w.r.t. last layer (mm)")
originalSize = get(gca, 'Position');
colorbar;
set(h, 'Position', originalSize);
set(gca,'fontname','times');set(gca,'FontSize',12);

% Display texture predictions
h0 = subplot(2,3,6);
imagesc(max(photo, cat(3, borderMap, borderMap, borderMap))); axis equal; ylim([1 yLim]);
title("Texture classification")
% Create colormap for classes
labelNumbers= {'0', '1', '2', '3'};
labels = {'fluid', 'good','dry','tear'};
colors = {[0 1 1], [0 1 0], [1 1 0], [1 0 0]};
numberDict = containers.Map(labelNumbers, labels);
colorDict = containers.Map(labels, colors);
% Plot classes per window
textures = -1 * ones(size(img));
for i=1:length(windowNames)
   window = windowNames(i);
   probs = table2array(textureTable(window, :));
   prob = max(probs);
   labelIdx = find(probs == prob);
   label = textureTable.Properties.VariableDescriptions{labelIdx(1)};
   textures(uint32(textCoordinates(i, 2)), uint32(textCoordinates(i, 1))) = str2num(label);
   label = numberDict(label);
   color = colorDict(label);
   label = strrep(label, '_', '-\n');
   text(textCoordinates(i, 1), textCoordinates(i, 2), sprintf("%s:\n%0.2f", sprintf(label), prob), 'Color', color, 'FontSize', 14, 'FontWeight', 'normal', 'FontName', 'times');
end
set(gca,'fontname','times');set(gca,'FontSize',12);

h1 = subplot(2,3,2);
% Create colormap to highlight defects (blue, red) and normality (green)
keyPoints = [orientationEdges(1) orientationThresholds(1) mean(orientationThresholds) orientationThresholds(end) orientationEdges(end)];
keyPoints = 1 + ceil(99 * (keyPoints - keyPoints(1)) / (keyPoints(end)-keyPoints(1)));
keyPoints = [keyPoints(1) keyPoints(2)-1 keyPoints(2:4) keyPoints(4)+1 keyPoints(5)];
myMap = interp2(X(keyPoints,:),Y(keyPoints,:),myMapColors,X,Y); %// interpolate colormap
% Plot
imagesc(thetaPlot,'AlphaData',thetaAlpha); caxis([orientationEdges(1) orientationEdges(end)]); colormap(gca, myMap); axis equal; ylim([1 yLim]);set(gca,'color','black');
title("Local orientation (degrees)")
originalSize = get(gca, 'Position');
colorbar;
set(h1, 'Position', originalSize);
thetas = theta;
thetas(skeleton == 0) = NaN;
set(gca,'fontname','times');set(gca,'FontSize',12);


subplot(2,3,1)
imagesc(photo), hold on; axis equal; ylim([1 yLim]);
title("Input image")
set(gca,'fontname','times');set(gca,'FontSize',12);

h2 = subplot(2,3,4);
% Create colormap to highlight defects (blue, red) and normality (green)
keyPoints = [thicknessEdges(1) thicknessThresholds(1) mean(thicknessThresholds) thicknessThresholds(end) thicknessEdges(end)];
keyPoints = 1 + ceil(99 * (keyPoints - keyPoints(1)) / (keyPoints(end)-keyPoints(1)));
keyPoints = [keyPoints(1) keyPoints(2)-1 keyPoints(2:4) keyPoints(4)+1 keyPoints(5)];
myMap = interp2(X(keyPoints,:),Y(keyPoints,:),myMapColors,X,Y); %// interpolate colormap
linearRelation = inv([thicknessEdges(1) 1; thicknessEdges(end) 1])*[1; length(myMap)]; % Create a linear function relating the width to the position in the color map
dRgb = ind2rgb(round(linearRelation(1) * disPlot + linearRelation(2)), myMap); % Convert the distanace map to RGB image
dRgb = min(dRgb, cat(3, disAlpha,disAlpha,disAlpha)); % Converting no-line regions to black
lines = im2double(img);
dRgb = max(dRgb, cat(3, lines, lines, lines)); % Adding interstitial lines in white
ax = gca;imagesc(dRgb); caxis(ax, [thicknessEdges(1) thicknessEdges(end)]); colormap(gca, myMap); axis equal; ylim([1 yLim]);set(gca,'color','black');
title("Local width (mm)")
originalSize = get(gca, 'Position');
colorbar;
set(h2, 'Position', originalSize);
widths = 2*realDistances;
widths(filteredSkeleton == 0) = NaN;
set(gca,'fontname','times');set(gca,'FontSize',12);

h3 = subplot(2,3,3);
% Create colormap to highlight defects (blue, red) and normality (green)
keyPoints = [curvatureEdges(1) curvatureThresholds(1) mean(curvatureThresholds) curvatureThresholds(end) curvatureEdges(end)];
keyPoints = 1 + ceil(99 * (keyPoints - keyPoints(1)) / (keyPoints(end)-keyPoints(1)));
keyPoints = [keyPoints(1) keyPoints(2)-1 keyPoints(2:4) keyPoints(4)+1 keyPoints(5)];
myMap = interp2(X(keyPoints,:),Y(keyPoints,:),myMapColors,X,Y); %// interpolate colormap
ax = gca;imagesc(curPlot,'AlphaData',curAlpha); caxis(ax, [curvatureEdges(1) curvatureEdges(end)]); colormap(gca, myMap); axis equal; ylim([1 yLim]);set(gca,'color','black');
title("Local curvature (mm^{-1})")
originalSize = get(gca, 'Position');
colorbar;
set(h3, 'Position', originalSize);
curvatures = kMap;
curvatures(kMap == 0) = NaN;
set(gca,'fontname','times');set(gca,'FontSize',12);

%%%

%%% Display histograms
set(0, 'currentfigure', figures(2)); 
clf(figures(2),'reset')
set(gcf,'name',"Histograms",'numbertitle','off')
if ~strcmp(figures(2).WindowState,'maximized')
    figures(2).WindowState='maximized';
end
subplot(2,3,1)
imagesc(photo), hold on; axis equal; ylim([1 yLim]); axis off;
title("Input image")
set(gca,'fontname','times');set(gca,'FontSize',12);

subplot(2,3,2);
orientation_hist = histogram(thetas(:), orientationEdges, 'Normalization', 'probability', 'FaceColor', 'cyan'); hold on;
plot(0.5*orientation_hist.BinWidth + orientation_hist.BinEdges(1:length(orientation_hist.BinEdges)-1), orientation_hist.Values, 'blue');
title("Orientation distribution");
ylim([0, 2e-3]);
xline(orientationThresholds(1), '--g', 'Color', [0 0.5 0], 'LineWidth', 2.0);
xline(orientationThresholds(2), '--g', 'Color', [0 0.5 0], 'LineWidth', 2.0);
set(gca,'fontname','times');set(gca,'FontSize',12);

subplot(2,3,3);
curvature_hist = histogram(curvatures(:), curvatureEdges, 'Normalization', 'probability', 'FaceColor', 'cyan'); hold on;
plot(0.5*curvature_hist.BinWidth + curvature_hist.BinEdges(1:length(curvature_hist.BinEdges)-1), curvature_hist.Values, 'blue');
title("Curvature distribution");
ylim([0, 1.2e-3]);
xline(curvatureThresholds(1), '--g', 'Color', [0 0.5 0], 'LineWidth', 2.0);
xline(curvatureThresholds(2), '--g', 'Color', [0 0.5 0], 'LineWidth', 2.0);
set(gca,'fontname','times');set(gca,'FontSize',12);

subplot(2,3,4);
width_hist = histogram(widths(:), thicknessEdges, 'Normalization', 'probability', 'FaceColor', 'cyan'); hold on;
plot(0.5*width_hist.BinWidth + width_hist.BinEdges(1:length(width_hist.BinEdges)-1), width_hist.Values, 'blue');
title("Width distribution");
ylim([0, 1.2e-3]);
xline(thicknessThresholds(1), '--g', 'Color', [0 0.5 0], 'LineWidth', 2.0);
xline(thicknessThresholds(2), '--g', 'Color', [0 0.5 0], 'LineWidth', 2.0);
set(gca,'fontname','times');set(gca,'FontSize',12);

subplot(2,3,5);
height_hist = histogram(nozzleHeights(:), heightEdges, 'Normalization', 'probability', 'FaceColor', 'cyan'); hold on;
plot(0.5*height_hist.BinWidth + height_hist.BinEdges(1:length(height_hist.BinEdges)-1), height_hist.Values, 'blue');
title("Nozzle height distribution");
ylim([0, 0.5]);
xline(heightThresholds(1), '--g', 'Color', [0 0.5 0], 'LineWidth', 2.0);
xline(heightThresholds(2), '--g', 'Color', [0 0.5 0], 'LineWidth', 2.0);
set(gca,'fontname','times');set(gca,'FontSize',12);

subplot(2,3,6);
[n,v] = groupcounts(textures(:));
n = n(2:end);
v = v(2:end);
textureClassNumbers = 0:str2double(labelNumbers(end));
texture_hist = [textureClassNumbers' zeros(length(textureClassNumbers), 1)];
for classNumber = textureClassNumbers
    try
        texture_hist(classNumber+1, 2) = n(v==classNumber);
    catch
        warning(sprintf("No instances of class '%s' found.", numberDict(num2str(classNumber))))
    end
end
bar(reordercats(categorical(labels), labels),texture_hist(:,2)/sum(n(:)), 'cyan'); hold on;
plot(reordercats(categorical(labels), labels),texture_hist(:,2)/sum(n(:)), 'blue');
title("Texture windows distribution");
ylim([0, 1.0]);
% xline(0.5, '--g', 'Color', [0 0.5 0], 'LineWidth', 2.0); hold on;
% xline(1.5, '--g', 'Color', [0 0.5 0], 'LineWidth', 2.0);
set(gca,'fontname','times');set(gca,'FontSize',12);

%%%

end
