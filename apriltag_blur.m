function generate_apriltag_blur


outputFolderName = 'generated_matlab'; 
exposureMs = 20;
samplesPerExposure = 81; %% # of snapshots simulated
linearTravelPx = [8 16 32]; %% total linear movement 
rotationSweepDeg = [6 12 24]; %% total rotational angle swept 

%%Inital roll and tilt angles of april tag (reference photo)
midpointRollDeg = 15;
midpointTiltDeg = 25;

%% validates all input combinations
validateattributes(exposureMs, {'numeric'}, {'scalar','real','finite','>=',0.001});
validateattributes(samplesPerExposure, {'numeric'}, {'scalar','integer','>=',2});
validateattributes(linearTravelPx, {'numeric'}, {'vector','nonempty','integer','>=',1,'<=',250});
validateattributes(rotationSweepDeg, {'numeric'}, {'vector','nonempty','integer','>=',1,'<=',90});
validateattributes(midpointRollDeg, {'numeric'}, {'scalar','real','finite'});
validateattributes(midpointTiltDeg, {'numeric'}, {'scalar','real','>=',-40,'<=',40});

%Original github where april tag was taken.
sourceUrl = ['https://raw.githubusercontent.com/AprilRobotics/apriltag-imgs/' ...
    'master/tag36h11/tag36_11_00000.png'];

%Rereating April Tag in binary matrix
tagCells = [
    1 1 1 1 1 1 1 1 1 1
    1 0 0 0 0 0 0 0 0 1
    1 0 1 1 0 1 0 1 0 1
    1 0 0 1 1 1 0 1 0 1
    1 0 0 1 1 0 0 0 0 1
    1 0 1 0 1 0 0 0 0 1
    1 0 0 1 0 1 1 0 0 1
    1 0 0 0 0 1 0 0 0 1
    1 0 0 0 0 0 0 0 0 1
    1 1 1 1 1 1 1 1 1 1
];

%%sizing
pixelsPerCell = 32;
tagTexture = repelem(255 * tagCells, pixelsPerCell, pixelsPerCell);
tagSideM = 0.12;  % Side length across the black square
tagCornersM = (tagSideM / 2) * [-1 -1 0; 1 -1 0; 1 1 0; -1 1 0];
tagCenterM = [0; 0; 0.35];
texturePixelsPerMeter = 8 * pixelsPerCell / tagSideM;

width = 640;
height = 480;
focalLengthPx = 600;

%% initializes virtual camera's focal length, and center of image
K = [focalLengthPx 0 (width-1)/2; 0 focalLengthPx (height-1)/2; 0 0 1];

%%reads vertical and horizontal coordinates and creates 2D matrx 
[u, v] = meshgrid(0:width-1, 0:height-1);


%finds viewing direction through each pixel 
cameraRays = K \ [u(:).'; v(:).'; ones(1, numel(u))];

%tilts and rotates tag
midpointRotation = rotationZ(midpointRollDeg) * rotationY(midpointTiltDeg);

%takes physical inputs declared earlier to and creates function to render

render = @(R) renderTag(tagTexture, texturePixelsPerMeter, R, ...
    tagCenterM, cameraRays, [height width]);


front = render(eye(3)); %renders straight image
sharp = render(midpointRotation); %renders image with tilt 




%%set up output folder
scriptFolder = fileparts(mfilename('fullpath'));
outputFolder = fullfile(scriptFolder, outputFolderName);
requireNewFolder(outputFolder);

%writes to a temporary folder until all images load
stagingFolder = tempname(scriptFolder);
[madeFolder, message] = mkdir(stagingFolder);
assert(madeFolder, '%s', message);
stagingCleanup = onCleanup(@() removeStagingFolder(stagingFolder)); 


%saves straight on view 
referenceName = 'sharp_midpoint.png';
static = struct('type', 'static');
saveImage = @(name, image, R, motion, reference) saveFrame( ...
    stagingFolder, name, image, R, motion, reference, tagCenterM, tagCornersM, K);
records = saveImage('sharp_front.png', front, eye(3), static, 'sharp_front.png');
records(end+1) = saveImage(referenceName, sharp, midpointRotation, static, referenceName);


phases = ((1:samplesPerExposure) - 0.5) / samplesPerExposure - 0.5;
exposureSeconds = exposureMs / 1000;
linearTravelPx = unique(linearTravelPx(:)).';
for travel = linearTravelPx
    shiftedFrame = @(phase) interp2(sharp, u + 1 - phase * travel, v + 1, 'linear', 255);
    blurred = averageExposure(shiftedFrame, phases, height, width);
    motion = struct('type', 'image_plane_translation_x', ...
        'travel_px', travel, 'speed_px_per_s', travel / exposureSeconds);
    name = sprintf('linear_%03dpx.png', travel);
    records(end+1) = saveImage(name, blurred, midpointRotation, motion, referenceName); %#ok<AGROW>
end

axisNames = {'roll', 'tilt'};
rotationFunctions = {@rotationZ, @rotationY};
cameraAxes = [0 0 1; 0 1 0];
rotationSweepDeg = unique(rotationSweepDeg(:)).';
for axisIndex = 1:numel(axisNames)
    axisName = axisNames{axisIndex};
    rotate = rotationFunctions{axisIndex};
    for sweep = rotationSweepDeg
        movingFrame = @(phase) render(rotate(phase * sweep) * midpointRotation);
        blurred = averageExposure(movingFrame, phases, height, width);
        motion = struct('type', ['rotation_' axisName '_about_tag_center'], ...
            'axis_in_camera_coordinates', cameraAxes(axisIndex,:), ...
            'sweep_deg', sweep, 'speed_deg_per_s', sweep / exposureSeconds, ...
            'speed_rad_per_s', (pi / 180) * sweep / exposureSeconds);
        name = sprintf('%s_%03ddeg.png', axisName, sweep);
        records(end+1) = saveImage(name, blurred, midpointRotation, motion, referenceName); %#ok<AGROW>
    end
end


truth.tag = struct('family', 'tag36h11', 'id', 0, ...
    'black_square_side_m', tagSideM, 'source_png', sourceUrl);
truth.camera = struct('width_px', width, 'height_px', height, 'K', K, ...
    'distortion_coefficients', zeros(1, 5), 'pixel_coordinates', 'zero-based');
truth.pose_convention = 'X_camera = R_camera_from_tag @ X_tag + t_camera_from_tag_m';
truth.tag_corner_order = 'top left, top right, bottom right, bottom left when upright';
truth.tag_corners_m = tagCornersM;
truth.exposure_ms = exposureMs;
truth.samples_per_blurred_image = samplesPerExposure;
truth.midpoint_roll_deg = midpointRollDeg;
truth.midpoint_tilt_deg = midpointTiltDeg;
truth.images = records;
writeGroundTruth(fullfile(stagingFolder, 'ground_truth.json'), truth);
requireNewFolder(outputFolder);
[movedFolder, message] = movefile(stagingFolder, outputFolder);
assert(movedFolder, '%s', message);
fprintf('Wrote %d PNGs and ground_truth.json to\n%s\n', numel(records), outputFolder);
end

function frame = renderTag(texture, pixelsPerMeter, R, t, rays, imageSize)
% Intersect each camera ray with the tag's plane, then sample the texture.
planePoints = [R(:,1:2), t] \ rays;
tagX = planePoints(1,:) ./ planePoints(3,:);
tagY = planePoints(2,:) ./ planePoints(3,:);
textureCenter = (size(texture, 1) + 1) / 2;
x = reshape(textureCenter + pixelsPerMeter * tagX, imageSize);
y = reshape(textureCenter + pixelsPerMeter * tagY, imageSize);
frame = interp2(texture, x, y, 'linear', 255);
end

function image = averageExposure(renderSample, phases, height, width)
image = zeros(height, width);
for phase = phases
    image = image + renderSample(phase);
end
image = image / numel(phases);
end

function entry = saveFrame(folder, name, image, R, motion, reference, t, cornersM, K)
imwrite(uint8(round(image)), fullfile(folder, name));
projected = K * (R * cornersM.' + t);
cornerPixels = (projected(1:2,:) ./ projected(3,:)).';
entry = struct('image', name, 'sharp_reference', reference, ...
    'R_camera_from_tag_midpoint', R, 't_camera_from_tag_midpoint_m', t.', ...
    'corners_midpoint_px', cornerPixels, 'motion', motion);
end

function R = rotationY(degrees)
c = cosd(degrees);
s = sind(degrees);
R = [c 0 s; 0 1 0; -s 0 c];
end

function R = rotationZ(degrees)
c = cosd(degrees);
s = sind(degrees);
R = [c -s 0; s c 0; 0 0 1];
end

function writeGroundTruth(path, truth)
file = fopen(path, 'wt');
assert(file ~= -1, 'Could not open %s for writing.', path);
fileCleanup = onCleanup(@() fclose(file)); 
text = jsonencode(truth);
written = fprintf(file, '%s\n', text);
assert(written == numel(text) + 1, 'Could not finish writing %s.', path);
end

function requireNewFolder(folder)
if exist(folder, 'file') ~= 0
    error('Output already exists at %s. Change outputFolderName for a new run.', folder);
end
end

function removeStagingFolder(folder)
if exist(folder, 'dir') == 7
    rmdir(folder, 's');
end
end
