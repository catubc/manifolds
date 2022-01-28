function process_ica_animal(sessions_filenames)

% function loads a list of session directories from a text file

fnames = strsplit(fileread(sessions_filenames));

% loop over each session
for i= 1:length(fnames)

    %
    filename = fnames(i);

    %
    filename = char(strcat(filename,'/suite2p/plane0/ensembles/F_upphase_binned_sum.npy'));
    
    %    
    [filepath, name, ext] = fileparts(filename);
    fname_out = strcat(filepath,'/Upphase_sum_binned_ICA_ASSEMBLIES.mat')
    if isfile(fname_out)
        clearvars -except fnames
        continue
    end
    
    filename
    % 
    DFF_binned = readNPY(filename);
    size(DFF_binned)

    % convert 
    CALCIUM_FLUORESCENCE_CONVERSION('','',DFF_binned, 0.5, 1)

    % run ICA and save to disk
    ICA_ASSEMBLY_DETECTION([ '/home/cat/_CALCIUM-FLUORESCENCE.mat' ] )

    % load ICA saved from disk in step above
    % TODO: try saving directly from ICA assembly detection step
    load('/home/cat/_ICA-ASSEMBLIES.mat')

    % copy results back to original directory
    save(fname_out)

    % clear all workspace
    clearvars -except fnames
end

