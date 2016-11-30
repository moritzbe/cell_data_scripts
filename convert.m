% List all files in folder
listing = dir('/Volumes/MoritzBertholdHD/CellData/Experiments/Ex1/TIF_images/Ex1_ch-PGP_rb-CGRP_mo-RIIb');

% Iterate through all files and look for DIBs
% for idx = 1:numel(listing)
% relevant files start at idx = 4, idx <4 gives filestructures
for idx = 4:numel(listing)
    filename = listing(idx);
    if ~isempty(strfind(filename.name,'o1.DIB')) 
       filename.name;
       pixels = readcdib(filename.name);
       idx
    end
end


function pixels = readcdib(filename)
    % Get the data for the whole DIB.
    fid = fopen(filename, 'r');
    buffer = fread(fid, inf, 'uint8=>uint8');
    fid = fclose(fid);

    % Extract the header elements.
    biSize = typecast(buffer(1:4), 'uint32');
    biWidth = typecast(buffer(5:8), 'uint32');
    biHeight = typecast(buffer(9:12), 'uint32');
    biPlanes = typecast(buffer(13:14), 'uint16');
    biBitCount = typecast(buffer(15:16), 'uint16');
    biCompression = typecast(buffer(17:20), 'uint32');
    biSizeImage = typecast(buffer(21:24), 'uint32');
    biXPelsPerMeter = typecast(buffer(25:28), 'uint32');
    biYPelsPerMeter = typecast(buffer(29:32), 'uint32');
    biClrUsed = typecast(buffer(33:36), 'uint32');
    biClrImportant = typecast(buffer(37:40), 'uint32');

    % Convert the pixels.
    startIdx = 1;
    endIdx = biWidth * biHeight * double(biBitCount) / 8;
    pixels = reshape(typecast(buffer(52 + (startIdx:endIdx)), 'uint16'), [biWidth biHeight]);
    
    % remove .DIB extension, and add new format
    filename = strcat(filename(1:end-4), '.TIF');
    imwrite(pixels, filename);
end



