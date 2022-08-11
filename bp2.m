%% �������ز� �����Ϊsigmod���� �����Ԫ�����ҲΪsigmod����
%% �������
clc;
clear;
close all;

%% ��������
load AB
% data=A_B1;

% data = A_B2;

data = A_B3;
load pre_data
[n,m] = size(data);
randindex = randperm(m);
new_data = data(: , randindex); %��������˳��

%% ����ѵ���Ͳ�������
train_num = round(0.8*m);
train_input = new_data(1:n-1 , 1:train_num);
train_output = new_data(n , 1:train_num);

test_input = new_data(1:n-1 , train_num+1:end);
test_output = new_data(n , train_num+1:end);

%% ��ʼ������
train_num; %ѵ��������
test_num = m - train_num; %����������
[~,pre_num] = size(pre_data) ;
cell_num = 6; %��Ԫ����
input_num = n-1; %���������
output_num = 1; %���������

learn_rate = 0.05; %ѧϰ��
Epochs_max = 50000; %����������
error_rate = 0.01; %Ŀ�����
Obj_save = zeros(1 , Epochs_max); %��ʧ����

miuij = 2*rand(cell_num , input_num) - 1; %���������ز��Ȩ��
theta_u = 2*rand(cell_num , 1) - 1; %���������ز��ƫ��
Omegaj = 2*rand(output_num , cell_num) - 1; %��������ز��Ȩ��
theta_y = 2*rand(output_num , 1) - 1; %��������ز��ƫ��

Epoch_errors = zeros(Epochs_max,1);

%% ��һ������
[normal_train_input,PS_train_input] = mapminmax(train_input, 0, 1);
[normal_train_output,PS_train_output] = mapminmax(train_output, 0, 1);

[normal_test_input,PS_test_input] = mapminmax(test_input, 0, 1);

[normal_pre_input,PS_pre_input] = mapminmax(pre_data, 0, 1);
%% ѵ������


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
    
    %�ݶ��½�
    c_Omegaj= 2*(pre_y - normal_train_output).*(1-pre_y).*pre_y*u';
    c_theta_y= 2*(pre_y - normal_train_output).*(1-pre_y).* pre_y*ones(train_num, 1);
    c_miuij=Omegaj'* 2*(pre_y - normal_train_output).*(1-pre_y).*(u).*(1-u).*pre_y* normal_train_input';
    c_theta_u=Omegaj'* 2*(pre_y - normal_train_output).*(1-pre_y).*(u).*(1-u).* pre_y*ones(train_num, 1);
    
    Omegaj=Omegaj-learn_rate*c_Omegaj;
    theta_y=theta_y-learn_rate*c_theta_y;  
    miuij=miuij- learn_rate*c_miuij;    
    theta_u=theta_u-learn_rate*c_theta_u;

end

%% ����
test_put = logsig(miuij*normal_test_input + repmat(theta_u, 1, test_num));
normal_test_output = logsig(Omegaj*test_put + repmat(theta_y, 1, test_num));

%% ����һ��
pre_test_output = mapminmax('reverse',normal_test_output,PS_train_output);

%% ���Լ����
errors = sum(abs((pre_test_output-test_output)./test_output))/length(test_output);

%% Ԥ��
pre_put =logsig(miuij*normal_pre_input + repmat(theta_u, 1, pre_num));
normal_pre_output = logsig(Omegaj*pre_put + repmat(theta_y, 1, pre_num));

pre_output = mapminmax('reverse',normal_pre_output,PS_train_output);
pre_output=pre_output';

     
%% �����ʾ
figure(1)
plot(Obj_save,'b-','LineWidth',1)
title('��ʧ�����仯')
xlabel('Epoch')
ylabel('Errors')

figure(2)
plot(test_output,'Color','b','LineWidth',1)
hold on
plot(pre_test_output,'*','Color','g')
hold on
title(['���ز�������Ԫ�������Ϊsigmod������BP������','   ���Ϊ��',num2str(errors),...
    '   ��������Ϊ��',num2str(epoch_num)])
legend('ԭ�����������ֵ','Ԥ������������ֵ')








































