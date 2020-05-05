clear;close all;clc;tic;
addpath('TestFunction')
addpath('Sampling Plans');
% number of repeat calculate
num_repeat = 30;
% initialize for repeat to save the data
num_addpoint = 100;   % number of iteration/addpoint
RepeatIGD = zeros(num_repeat, num_addpoint+1);    RepeatHV = zeros(num_repeat, num_addpoint+1);
RepeatPareto = cell(num_repeat,num_addpoint+1);   RepeatET = zeros(num_repeat, 1);
for iii = 1 : num_repeat
    fun_name = 'ZDT1';
    infill_name= 'LCBM_Euclidean';
    num_obj = 2;   num_vari = 6;   num_initial = 11*num_vari-1;
    % the maximum allowed evaluations
    max_evaluation = (num_addpoint + num_initial);
    switch fun_name
        case {'ZDT1', 'ZDT2', 'ZDT3'}
            design_space=[zeros(1,num_vari);ones(1,num_vari)]; ref_point = 1.2*high_PF(fun_name, num_obj);
        case {'DTLZ2','DTLZ5','DTLZ7'}
            design_space=[zeros(1,num_vari);ones(1,num_vari)]; ref_point = 1.2*high_PF(fun_name, num_obj);
    end
    sample_x = repmat(design_space(1,:),num_initial,1) + repmat(design_space(2,:)-design_space(1,:),num_initial,1).*bestlh(num_initial, num_vari, 20, 10);
    sample_y = feval(fun_name, sample_x, num_obj);
    % scale the objectives to [0,1]
    sample_y_scaled =(sample_y - repmat(min(sample_y),size(sample_y,1),1))./repmat(max(sample_y)-min(sample_y),size(sample_y,1),1);
    % initialize some parameters
    evaluation = size(sample_x,1);
    kriging_obj = cell(1,num_obj);
    iteration = 0;
    index = Paretoset(sample_y);
    non_dominated_front = sample_y(index,:);
    non_dominated_front_scaled = sample_y_scaled(index,:);
    %% HV
    hypervolume = zeros(max_evaluation - num_initial+1,1);
    hypervolume(1) = NorHV_Cal(non_dominated_front,ref_point);
    %% igd
    TruePareto = ParetoTrue(fun_name, num_vari, num_obj);
    igd = zeros(1,max_evaluation - num_initial+1);
    igd(1) = IGD(non_dominated_front, TruePareto);
    %% pareto
    RepeatPareto{iii,iteration + 1} = non_dominated_front;
    % beginning of the iteration
    t0 = tic;
    while evaluation < max_evaluation
        % build the initial kriging model for each objective
        for ii=1:num_obj
            kriging_obj{ii} = dacefit(sample_x,sample_y_scaled(:,ii),'regpoly0','corrgauss',1*ones(1,num_vari),0.001*ones(1,num_vari),1000*ones(1,num_vari));
        end
        % select updating points using the LCBM criteria
        switch infill_name
            case 'EI_Euclidean'
                infill_criterion = @(x)Infill_EI_Euclidean(x, kriging_obj, non_dominated_front_scaled);
            case 'LCBM_Euclidean'
                infill_criterion = @(x)Infill_LCBM_Euclidean(x, kriging_obj, non_dominated_front_scaled);
            case 'LCBM_Maximin'
                infill_criterion = @(x)Infill_LCBM_Maximin(x, kriging_obj, non_dominated_front_scaled);
            case 'LCBM_Hypervolume'
                infill_criterion = @(x)Infill_LCBM_Hypervolume(x, kriging_obj, non_dominated_front_scaled);
            case 'UHVI'
                infill_criterion = @(x)Infill_UHVI(x, kriging_obj, non_dominated_front_scaled);
        end
        % repeat calcute DE 4 times to get the best x
        for ii = 1:4
            [best4(ii,:), best_improvement(ii,:), ~] = DE(infill_criterion, num_vari, design_space(1,:), design_space(2,:), 50, 200);
        end
        [~, ind] = min(best_improvement);
        best_x = best4(ind, :);
        CloseDist = min(pdist2(best_x, sample_x));
        if CloseDist<10e-8
            % maximize the uncertainty, namely PI
            infill_criterion_uncertainty = @(x)Infill_Standard_Uncertainty(x, kriging_obj, non_dominated_front_scaled);
            best_x = DE(infill_criterion_uncertainty, num_vari, design_space(1,:), design_space(2,:), 50, 200);
        end
        % add the new points to the design set
        sample_x = [sample_x;best_x];
        sample_y = [sample_y; feval(fun_name,best_x, num_obj)];
        sample_y_scaled = (sample_y - repmat(min(sample_y),size(sample_y,1),1))./repmat(max(sample_y)-min(sample_y),size(sample_y,1),1);
        evaluation = evaluation + size(best_x,1);
        iteration = iteration + 1;
        % calculate the hypervolume values and print them on the screen
        index = Paretoset(sample_y);
        non_dominated_front = sample_y(index,:);
        non_dominated_front_scaled = sample_y_scaled(index,:);
        hypervolume(iteration + 1) = NorHV_Cal(non_dominated_front, ref_point);
        igd(iteration + 1) = IGD(non_dominated_front, TruePareto);
        RepeatPareto{iii, iteration + 1} = non_dominated_front;
        % plot current non-dominated front points
        if num_obj == 2
            scatter(non_dominated_front(:,1), non_dominated_front(:,2),80,'ro', 'filled');
            title(sprintf('iteration: %d, evaluations: %d',iteration,evaluation));drawnow;
        elseif num_obj == 3
            scatter3(non_dominated_front(:,1), non_dominated_front(:,2),non_dominated_front(:,3),80,'ro', 'filled');
            title(sprintf('iteration: %d, evaluations: %d',iteration,evaluation));drawnow;
        end
        % print the hypervolume information
        fprintf(' num_repeat: %d, iteration: %d, evaluation: %d, hypervolume: %f\n', iii, iteration, evaluation, hypervolume(iteration +1));
    end
    runtime=toc(t0);
    % plot true non-dominated front points
    hold on
    switch fun_name
        case {'ZDT1', 'ZDT2'}
            plot(TruePareto(:, 1), TruePareto(:, 2), 'k', 'linewidth',2)
        case {'ZDT3'}
            mm = TruePareto(:, 1);  nn = TruePareto(:, 2);
            plot(mm(mm<0.13), nn(mm<0.13), 'k', mm(0.13<mm&mm<0.32), nn(0.13<mm&mm<0.32), 'k',  mm(0.32<mm&mm<0.55), nn(0.32<mm&mm<0.55), 'k', ...
                mm(0.55<mm&mm<0.75), nn(0.55<mm&mm<0.75), 'k',  mm(0.75<mm), nn(0.75<mm), 'k', 'linewidth',2)
        case {'DTLZ2', 'DTLZ5', 'DTLZ7'}
            scatter3(TruePareto(:, 1), TruePareto(:, 2),TruePareto(:, 3),20, 'ko');
    end
    % save the data of pareto/HV/IGD
    RepeatHV(iii,:) = hypervolume';
    RepeatIGD(iii,:)=igd;
    RepeatET(iii) = runtime/60;
    close all;
end