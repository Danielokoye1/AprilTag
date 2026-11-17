clear;
clc;
close all;

% make the AprilTag
tag = [
    1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1
    1 1 0 0 0 0 0 0 0 0 1 1 
    1 1 0 1 1 0 1 0 1 0 1 1
    1 1 0 0 1 1 1 0 1 0 1 1
    1 1 0 0 1 1 0 0 0 0 1 1
    1 1 0 1 0 1 0 0 0 0 1 1
    1 1 0 0 1 0 1 1 0 0 1 1
    1 1 0 0 0 0 1 0 0 0 1 1
    1 1 0 0 0 0 0 0 0 0 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 
    1 1 1 1 1 1 1 1 1 1 1 1 ];

lengthOfMotions = [0 16 32];
rotateAngles = [6 12 24]; 


% enlarge pixels and create black and white pixels
pixelspercell = 40; 
tagImage = uint8(repelem(tag, pixelspercell, pixelspercell) * 255);

% Linear motion blur (horizontal, vertical, and Diagonal)
numSamples = 21;

% Horizontal linear blur
subplot(5,4,1);
imshow(tagImage);
title('Original Tag');

for i = 1:length(lengthOfMotions)
    dist = lengthOfMotions(i);
    blurred = zeros(size(tagImage));
    for shift = linspace(-dist/2, dist/2, numSamples) % add blurring to image
        shifted = imtranslate(tagImage, [shift 0], FillValues = 255); % shift the image
        blurred = blurred + double(shifted); % add shifted image to running total
    end
    blurred = uint8(blurred/numSamples); % average out shifted images creating blur
    subplot(5,4,i+1);
    imshow(blurred);
    title("Horizontal " + num2str(dist) + " pixles");
    imwrite(blurred, "horizontal_" + num2str(dist) + "pixel.png");
end

% Vertical linear blur (same as horizontal just different shift direction [0 shift])
subplot(5,4,5);
imshow(tagImage);
title('Original Tag');

for i = 1:length(lengthOfMotions)
    dist = lengthOfMotions(i);
    blurred = zeros(size(tagImage));
    for shift = linspace(-dist/2, dist/2, numSamples) % add blurring to image
        shifted = imtranslate(tagImage, [0 shift], FillValues = 255); % shift the image
        blurred = blurred + double(shifted); % add shifted image to running total
    end
    blurred = uint8(blurred/numSamples); % average out shifted images creating blur
    subplot(5,4,i+5);
    imshow(blurred);
    title("Vertical " + num2str(dist) + " pixles");
    imwrite(blurred, "vertical_" + num2str(dist) + "pixel.png");
end

% Diagonal linear blur (same as horizontal just different shift direction [shift shift])
subplot(5,4,9);
imshow(tagImage);
title('Original Tag');

for i = 1:length(lengthOfMotions)
    dist = lengthOfMotions(i);
    blurred = zeros(size(tagImage));
    for shift = linspace(-dist/2, dist/2, numSamples) % add blurring to image
        shifted = imtranslate(tagImage, [shift shift], FillValues = 255); % shift the image
        blurred = blurred + double(shifted); % add shifted image to running total
    end
    blurred = uint8(blurred/numSamples); % average out shifted images creating blur
    subplot(5,4,i+9);
    imshow(blurred);
    title("Diagonal " + num2str(dist) + " pixles");
    imwrite(blurred, "diagonal_" + num2str(dist) + "pixel.png");
end


% Opposite Diagonal linear blur (same as horizontal just different shift direction [shift -shift])
subplot(5,4,13);
imshow(tagImage);
title('Original Tag');

for i = 1:length(lengthOfMotions)
    dist = lengthOfMotions(i);
    blurred = zeros(size(tagImage));
    for shift = linspace(-dist/2, dist/2, numSamples) % add blurring to image
        shifted = imtranslate(tagImage, [shift -shift], FillValues = 255); % shift the image
        blurred = blurred + double(shifted); % add shifted image to running total
    end
    blurred = uint8(blurred/numSamples); % average out shifted images creating blur
    subplot(5,4,i+13);
    imshow(blurred);
    title("Opp Diagonal " + num2str(dist) + " pixles");
    imwrite(blurred, "opp_diagonal_" + num2str(dist) + "pixel.png");
end

% Rotational motion blur
subplot(5, 4, 17);
imshow(tagImage);
title('Original Tag');
for i = 1:length(rotateAngles)
    angle = rotateAngles(i);
    blurred = zeros(size(tagImage));
    for rotation = linspace(0, angle, numSamples) % Rotate AprilTag CCW
        rotated = imrotate(255 - double(tagImage), rotation, 'bilinear', 'crop'); % shift the image
        rotated = 255 - rotated;
        blurred = blurred + rotated; % add shifted image to running total

    end
    blurred = uint8(blurred/numSamples); % average out shifted images creating blur
    subplot(5,4,i+17);
    imshow(blurred);
    title("CCW Rotation " + num2str(angle) + " Degree Blur");
    imwrite(blurred, "rotation_CCW_" + num2str(angle) + "deg.png");
end

