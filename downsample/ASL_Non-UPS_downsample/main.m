clc;
close all;
clear;

% hyper parameter MaxNumEvents
max_points = 40;
res_root = 'D:\ycg\ASL_DVS';
des_root = 'D:\ycg\ASL_Nonuniform_downsample';

% delete old downsample directory


people = dir(res_root);
for p=1:length(people)

    if( isequal( people( p ).name, '.' )||...
        isequal( people( p ).name, '..')||...
        ~people( p ).isdir)               % skip other file
        label = -1;
        continue;
    end
    

    label = people(p).name;
    if(~isequal(label,'g'))
        continue;
    end
    if ~exist([des_root filesep people(p).name], 'dir')
            mkdir([des_root, filesep, people(p).name]);
    end

    count = dir([res_root filesep people( p ).name]);
    for j = 1:length(count)
        if( isequal( count(j).name, '.' )||...
            isequal( count(j).name, '..')||...
            count(j).isdir)                % skip other file
            continue;
        end

        data_path = [res_root, filesep, people(p).name, filesep, count(j).name];
        idx = j-3;
        save_path = [des_root, filesep, people(p).name, filesep, num2str(idx)];
        if exist([des_root, filesep, people(p).name, filesep, [num2str(idx),'.mat']] , 'file')
            continue
        end
        pcd_downsampler(data_path, save_path, max_points);
        % scene+" "+subject+" "+j
    end
end
fprintf("Finshed\n");
