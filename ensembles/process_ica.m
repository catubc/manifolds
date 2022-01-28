function process_ica(filename)

% 
DFF_binned = readNPY(filename);
size(DFF_binned)

% convert 
CALCIUM_FLUORESCENCE_CONVERSION('','',DFF_binned, 0.5, 1)

% run ICA and save to disk
ICA_ASSEMBLY_DETECTION([ '/home/cat/_CALCIUM-FLUORESCENCE.mat' ] )

% load ICA saved assemblies
load('/home/cat/_ICA-ASSEMBLIES.mat')

% copy results back to original directory
[filepath, name, ext] = fileparts(filename)
save(strcat(filepath,'/Upphase_sum_binned_ICA_ASSEMBLIES.mat'))

% clear all workspace
clear