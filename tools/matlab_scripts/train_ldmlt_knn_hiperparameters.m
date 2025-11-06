clear; clc; close all;

addpath('./LDMLT_TS');
DATA_PATH = '../../data/matlab_data_for_LDMLT.mat';
RESULTS_DIR_BASE = '../../data/training_results/ldmlt_knn_results';

load(DATA_PATH);

disp('>>> STARTING TEST HIPERPARAMETRS FOR LDMLT <<<');

MIN_SEQUENCE_LENGTH = 2;
valid_indices = cellfun(@(x) size(x, 1) >= MIN_SEQUENCE_LENGTH, sequences);
sequences_raw = sequences(valid_indices);
labels_raw = labels(valid_indices);

stroke_labels = cell(length(labels_raw), 1);
subjects = cell(length(labels_raw), 1);
for i = 1:length(labels_raw)
    parts = strsplit(labels_raw{i}, ' ');
    stroke_labels{i} = parts{1};
    subjects{i} = parts{2};
end

[y_numeric, unique_stroke_labels_cat] = grp2idx(categorical(stroke_labels));
unique_stroke_labels = cellstr(unique_stroke_labels_cat);

disp('Mapped labels to numbers:');
for i = 1:length(unique_stroke_labels)
    fprintf('  %d: %s\n', i, unique_stroke_labels{i});
end

unique_subjects = unique(subjects);
fprintf('\nFound %d unique subjects for LOSO CV:\n', length(unique_subjects));
disp(unique_subjects');


% Definition of the hyperparameter
k_values = [1, 3, 4, 5, 7, 9, 11, 13, 15];
tripletsfactor_values = [5, 10, 15, 20, 25, 30, 40, 50];
cycle_values = [5, 8, 10, 15, 20, 25];
alphafactor_values = [1, 2, 4, 5, 6, 8, 10];

best_params = struct();
best_params.k = 1;
best_params.tripletsfactor = 20;
best_params.cycle = 15;
best_params.alphafactor = 5;

tuning_results = struct();

% Tuning K_vector
current_params = best_params;
accuracies_k = [];
for k_val = k_values
    fprintf('\n Testing k = %d \n', k_val);
    mean_acc = run_loso_validation(sequences_raw, subjects, y_numeric, unique_subjects, k_val, current_params);
    accuracies_k(end+1) = mean_acc;
end
[max_acc_k, best_idx_k] = max(accuracies_k);
best_params.k = k_values(best_idx_k);
tuning_results.k_tuning = table(k_values', accuracies_k', 'VariableNames', {'k', 'Accuracy'});
fprintf('\nBest k = %d (Accuracy: %.4f)\n', best_params.k, max_acc_k);
disp(tuning_results.k_tuning);

% Tuning tripletsfactor
current_params = best_params;
accuracies_tf = [];
for tf_val = tripletsfactor_values
    fprintf('\n Testing tripletsfactor = %d \n', tf_val);
    current_params.tripletsfactor = tf_val;
    mean_acc = run_loso_validation(sequences_raw, subjects, y_numeric, unique_subjects, best_params.k, current_params);
    accuracies_tf(end+1) = mean_acc;
end
[max_acc_tf, best_idx_tf] = max(accuracies_tf);
best_params.tripletsfactor = tripletsfactor_values(best_idx_tf);
tuning_results.tripletsfactor_tuning = table(tripletsfactor_values', accuracies_tf', 'VariableNames', {'tripletsfactor', 'Accuracy'});
fprintf('\nBest tripletsfactor = %d (Accuracy: %.4f)\n', best_params.tripletsfactor, max_acc_tf);
disp(tuning_results.tripletsfactor_tuning);

% Tuning cycle
current_params = best_params;
accuracies_cy = [];
for cy_val = cycle_values
    fprintf('\n Testing cycle = %d \n', cy_val);
    current_params.cycle = cy_val;
    mean_acc = run_loso_validation(sequences_raw, subjects, y_numeric, unique_subjects, best_params.k, current_params);
    accuracies_cy(end+1) = mean_acc;
end
[max_acc_cy, best_idx_cy] = max(accuracies_cy);
best_params.cycle = cycle_values(best_idx_cy);
tuning_results.cycle_tuning = table(cycle_values', accuracies_cy', 'VariableNames', {'cycle', 'Accuracy'});
fprintf('\nBest cycle = %d (Accuracy: %.4f)\n', best_params.cycle, max_acc_cy);
disp(tuning_results.cycle_tuning);

% Tuning alphafactor
current_params = best_params;
accuracies_af = [];
for af_val = alphafactor_values
    fprintf('\nTesting alphafactor = %d \n', af_val);
    current_params.alphafactor = af_val;
    mean_acc = run_loso_validation(sequences_raw, subjects, y_numeric, unique_subjects, best_params.k, current_params);
    accuracies_af(end+1) = mean_acc;
end
[max_acc_af, best_idx_af] = max(accuracies_af);
best_params.alphafactor = alphafactor_values(best_idx_af);
tuning_results.alphafactor_tuning = table(alphafactor_values', accuracies_af', 'VariableNames', {'alphafactor', 'Accuracy'});
fprintf('\nBest alphafactor = %d (Accuracy: %.4f)\n', best_params.alphafactor, max_acc_af);
disp(tuning_results.alphafactor_tuning);

fprintf('Best found parameters:\n');
disp(best_params);

timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
RESULTS_DIR = fullfile(RESULTS_DIR_BASE, sprintf('%s_tuning_results', timestamp));
if ~exist(RESULTS_DIR, 'dir'), mkdir(RESULTS_DIR); end

save(fullfile(RESULTS_DIR, 'tuning_results.mat'), 'tuning_results');
writetable(tuning_results.k_tuning, fullfile(RESULTS_DIR, 'tuning_k.csv'));
writetable(tuning_results.tripletsfactor_tuning, fullfile(RESULTS_DIR, 'tuning_tripletsfactor.csv'));
writetable(tuning_results.cycle_tuning, fullfile(RESULTS_DIR, 'tuning_cycle.csv'));
writetable(tuning_results.alphafactor_tuning, fullfile(RESULTS_DIR, 'tuning_alphafactor.csv'));

fid = fopen(fullfile(RESULTS_DIR, 'best_params.txt'), 'w');
fprintf(fid, 'Best k: %d\n', best_params.k);
fprintf(fid, 'Best tripletsfactor: %d\n', best_params.tripletsfactor);
fprintf(fid, 'Best cycle: %d\n', best_params.cycle);
fprintf(fid, 'Best alphafactor: %d\n', best_params.alphafactor);
fclose(fid);

disp('>>> FINISHED <<<');


function mean_accuracy = run_loso_validation(sequences, subjects, y_numeric, unique_subjects, K, ldmlt_params)
    fold_accuracies = [];
    for i = 1:length(unique_subjects)
        subject_to_leave_out = unique_subjects{i};
        fprintf('\nTesting on Subject: %s\n', subject_to_leave_out);

        test_mask = strcmp(subjects, subject_to_leave_out);
        train_mask = ~test_mask;
        
        X_train = sequences(train_mask)';
        y_train = y_numeric(train_mask);
        
        X_test = sequences(test_mask)';
        y_test = y_numeric(test_mask);
        
        M = LDMLT_TS(X_train, y_train', ldmlt_params);
        
        Pred_Y = KNN_TS(X_train, y_train, X_test, M, K);
        
        y_pred = Pred_Y';
        
        fold_accuracy = sum(y_pred == y_test) / length(y_test);
        fold_accuracies(end+1) = fold_accuracy;
        fprintf('Accuracy for subject %s: %.4f\n', subject_to_leave_out, fold_accuracy);
    end
    mean_accuracy = mean(fold_accuracies);
end