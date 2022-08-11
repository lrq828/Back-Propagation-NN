clear
clc
%% 加载数据集
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
%% 随机划分数据集
temp=randperm(inputNum);
P_train=P(:,temp(1:0.8*inputNum));
T_train=T(:,temp(1:0.8*inputNum));
P_test=P(:,temp(0.8*inputNum+1:end));
T_test=T(:,temp(0.8*inputNum+1:end));
%% 归一化
[p_train,ps_train]=mapminmax(P_train,0,1);
p_test=mapminmax('apply',P_test,ps_train);
[t_train,ps_output]=mapminmax(T_train,0,1);
%% 搭建网络
net=newff(p_train,t_train,[hiddennum,hiddennum]);

net.trainParam.epochs = 50000;
net.trainParam.goal=1e-5;
net.trainParam.lr=0.05;
%% 开始训练
net = train(net,p_train,t_train);

%% 测试网络
t_sim = sim(net,p_test);
T_sim = mapminmax('reverse',t_sim,ps_output);
err = norm(T_sim-T_test);

%% 查看训练后的权值和阈值
w1=net.iw{1,1}; %输入层到隐含层的权值
b1=net.b{1};    %隐含层神经元阈值
w2=net.iw{2,1}; %隐含层到输出层的权值
b2=net.b{2};    %输出层阈值
iw=net.iw;  %各层间权值
b=net.b;  %各层阈值

%% 预测 
%B预测C
% Pre_data=[29.576,83.477,202.331 ]';
% pre_input=mapminmax('apply',Pre_data,ps_train);
% pre_out=sim(net,pre_input);
% Pre_output = mapminmax('reverse',pre_out,ps_output);

% C2预测C1C3
% Pre_data=[83.477]';
% pre_input=mapminmax('apply',Pre_data,ps_train);
% pre_out=sim(net,pre_input);
% Pre_output = mapminmax('reverse',pre_out,ps_output);

% B预测A
Pre_data=[2.474,95.953,84.641]';
pre_input=mapminmax('apply',Pre_data,ps_train);
pre_out=sim(net,pre_input);
Pre_output = mapminmax('reverse',pre_out,ps_output);
