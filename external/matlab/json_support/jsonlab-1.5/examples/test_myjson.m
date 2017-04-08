close all;
clear;
clc;

fname='dqn_SpaceInvaders-v0_log_dqn_deep.json';
% json=savejson('data',loadjson(fname));
% fprintf(1,'%s\n',json);
% fprintf(1,'%s\n',savejson('data',loadjson(fname),'Compact',1));
data=loadjson(fname);
% savejson('data',data,'selftest.json');
% data=loadjson('selftest.json');