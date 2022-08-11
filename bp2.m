%% 单层隐藏层 激活函数为sigmod函数 输出神经元激活函数也为sigmod函数
%% 清除界面
clc;
clear;
close all;

%% 导入数据
load AB
% data=A_B1;

% data = A_B2;

data = A_B3;
load pre_data
[n,m] = size(data);
randindex = randperm(m);
new_data = data(: , randindex); %打乱样本顺序

%% 设置训练和测试数据
train_num = round(0.8*m);
train_input = new_data(1:n-1 , 1:train_num);
train_output = new_data(n , 1:train_num);

test_input = new_data(1:n-1 , train_num+1:end);
test_output = new_data(n , train_num+1:end);

%% 初始化参数
train_num; %训练样本数
test_num = m - train_num; %测试样本数
[~,pre_num] = size(pre_data) ;
cell_num = 6; %神经元个数
input_num = n-1; %输入变量数
output_num = 1; %输出变量数

learn_rate = 0.05; %学习率
Epochs_max = 50000; %最大迭代次数
error_rate = 0.01; %目标误差
Obj_save = zeros(1 , Epochs_max); %损失函数

miuij = 2*rand(cell_num , input_num) - 1; %输入与隐藏层的权重
theta_u = 2*rand(cell_num , 1) - 1; %输入与隐藏层的偏置
Omegaj = 2*rand(output_num , cell_num) - 1; %输出与隐藏层的权重
theta_y = 2*rand(output_num , 1) - 1; %输出与隐藏层的偏置

Epoch_errors = zeros(Epochs_max,1);

%% 归一化处理
[normal_train_input,PS_train_input] = mapminmax(train_input, 0, 1);
[normal_train_output,PS_train_output] = mapminmax(train_output, 0, 1);

[normal_test_input,PS_test_input] = mapminmax(test_input, 0, 1);

[normal_pre_input,PS_pre_input] = mapminmax(pre_data, 0, 1);
%% 训练网络


epoch_num = 0;
while epoch_num <= Epochs_max
    epoch_num = epoch_num + 1;
    
    P=miuij * normal_train_input + repmat(theta_u, 1, train_num);
    u = logsig(P);
  
    G = Omegaj * u + repmat(theta_y, 1, train_num);
    pre_y = logsig(G);
    
    obj = pre_y - normal_train_output;
    
    Ems = sumsqr(obj);
    Obj_save(epoch_num) = Ems;

    if Ems < error_rate
        break;
    end
    
    %梯度下降
    c_Omegaj= 2*(pre_y - normal_train_output).*(1-pre_y).*pre_y*u';
    c_theta_y= 2*(pre_y - normal_train_output).*(1-pre_y).* pre_y*ones(train_num, 1);
    c_miuij=Omegaj'* 2*(pre_y - normal_train_output).*(1-pre_y).*(u).*(1-u).*pre_y* normal_train_input';
    c_theta_u=Omegaj'* 2*(pre_y - normal_train_output).*(1-pre_y).*(u).*(1-u).* pre_y*ones(train_num, 1);
    
    Omegaj=Omegaj-learn_rate*c_Omegaj;
    theta_y=theta_y-learn_rate*c_theta_y;  
    miuij=miuij- learn_rate*c_miuij;    
    theta_u=theta_u-learn_rate*c_theta_u;

end

%% 测试
test_put = logsig(miuij*normal_test_input + repmat(theta_u, 1, test_num));
normal_test_output = logsig(Omegaj*test_put + repmat(theta_y, 1, test_num));

%% 反归一化
pre_test_output = mapminmax('reverse',normal_test_output,PS_train_output);

%% 测试集误差
errors = sum(abs((pre_test_output-test_output)./test_output))/length(test_output);

%% 预测
pre_put =logsig(miuij*normal_pre_input + repmat(theta_u, 1, pre_num));
normal_pre_output = logsig(Omegaj*pre_put + repmat(theta_y, 1, pre_num));

pre_output = mapminmax('reverse',normal_pre_output,PS_train_output);
pre_output=pre_output';

     
%% 结果显示
figure(1)
plot(Obj_save,'b-','LineWidth',1)
title('损失函数变化')
xlabel('Epoch')
ylabel('Errors')

figure(2)
plot(test_output,'Color','b','LineWidth',1)
hold on
plot(pre_test_output,'*','Color','g')
hold on
title(['隐藏层和输出神经元激活函数均为sigmod函数的BP神经网络','   误差为：',num2str(errors),...
    '   迭代次数为：',num2str(epoch_num)])
legend('原测试样本输出值','预测测试样本输出值')








































