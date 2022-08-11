clear
clc
%% �������ݼ�
% load BC_data
%C->B
% T=[B1,B2,B3]';
% P=[C1,C2,C3]';

%B1->B2,B3
% P=[B1]';
% T=[B2,B3]';

%C2->C1,C3
% P=[C2]';
% T=[C1,C3]';

%B->A
load AB
T=[A_B1(1:2,:)];
P=[A_B1(3,:);A_B2(3,:);A_B3(3,:)];

inputDin=size(P,1);
inputNum=size(P,2);
inputnum=size(P,1);
hiddennum=2*inputnum+1;
outputnum=size(T,1);
%% ����������ݼ�
temp=randperm(inputNum);
P_train=P(:,temp(1:0.8*inputNum));
T_train=T(:,temp(1:0.8*inputNum));
P_test=P(:,temp(0.8*inputNum+1:end));
T_test=T(:,temp(0.8*inputNum+1:end));
%% ��һ��
[p_train,ps_train]=mapminmax(P_train,0,1);
p_test=mapminmax('apply',P_test,ps_train);
[t_train,ps_output]=mapminmax(T_train,0,1);
%% �����
net=newff(p_train,t_train,[hiddennum,hiddennum]);

net.trainParam.epochs = 50000;
net.trainParam.goal=1e-5;
net.trainParam.lr=0.05;
%% ��ʼѵ��
net = train(net,p_train,t_train);

%% ��������
t_sim = sim(net,p_test);
T_sim = mapminmax('reverse',t_sim,ps_output);
err = norm(T_sim-T_test);

%% �鿴ѵ�����Ȩֵ����ֵ
w1=net.iw{1,1}; %����㵽�������Ȩֵ
b1=net.b{1};    %��������Ԫ��ֵ
w2=net.iw{2,1}; %�����㵽������Ȩֵ
b2=net.b{2};    %�������ֵ
iw=net.iw;  %�����Ȩֵ
b=net.b;  %������ֵ

%% Ԥ�� 
%BԤ��C
% Pre_data=[29.576,83.477,202.331 ]';
% pre_input=mapminmax('apply',Pre_data,ps_train);
% pre_out=sim(net,pre_input);
% Pre_output = mapminmax('reverse',pre_out,ps_output);

% C2Ԥ��C1C3
% Pre_data=[83.477]';
% pre_input=mapminmax('apply',Pre_data,ps_train);
% pre_out=sim(net,pre_input);
% Pre_output = mapminmax('reverse',pre_out,ps_output);

% BԤ��A
Pre_data=[2.474,95.953,84.641]';
pre_input=mapminmax('apply',Pre_data,ps_train);
pre_out=sim(net,pre_input);
Pre_output = mapminmax('reverse',pre_out,ps_output);
