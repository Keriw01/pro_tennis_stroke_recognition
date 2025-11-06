clear; clc; close all;

addpath('./LDMLT_TS');
DATA_PATH = '../../data/matlab_data_for_LDMLT.mat';
RESULTS_DIR_BASE = '../../data/training_results/ldmlt_knn_results';

load(DATA_PATH);

disp('>>> STARTING LDMLT + k-NN WITH LOSO CV <<<');

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


disp('Starting Leave-One-Subject-Out Cross-Validation...');
K_vector = 11;
params = struct('tripletsfactor', 50, 'cycle', 20, 'alphafactor', 5);

fold_accuracies = containers.Map('KeyType', 'char', 'ValueType', 'double');
fold_classification_times = containers.Map('KeyType', 'char', 'ValueType', 'double');
all_y_test = [];
all_y_pred = [];

for i = 1:length(unique_subjects)
    subject_to_leave_out = unique_subjects{i};
    fprintf('\nTesting on Subject: %s\n', subject_to_leave_out);
    
    test_mask = strcmp(subjects, subject_to_leave_out);
    train_mask = ~test_mask;
    
    X_train = sequences_raw(train_mask)';
    y_train = y_numeric(train_mask);
    
    X_test = sequences_raw(test_mask)';
    y_test = y_numeric(test_mask);
    
    fprintf('Training set size: %d, Test set size: %d\n', length(X_train), length(X_test));
    
    M = LDMLT_TS(X_train, y_train', params);
    
    tic;
    Pred_Y = KNN_TS(X_train, y_train, X_test, M, K_vector);
    classification_time = toc;
    
    y_pred = Pred_Y';
    
    avg_time_per_sample = classification_time / length(X_test);
    fprintf('Classification completed in %.4f seconds (avarage %.6f s/sample).\n', classification_time, avg_time_per_sample);
    
    fold_accuracy = sum(y_pred == y_test) / length(y_test);
    fold_accuracies(subject_to_leave_out) = fold_accuracy;
    fold_classification_times(subject_to_leave_out) = avg_time_per_sample;
    
    all_y_test = [all_y_test; y_test];
    all_y_pred = [all_y_pred; y_pred];
    
    fprintf('Accuracy for subject %s: %.4f\n', subject_to_leave_out, fold_accuracy);
end

accuracies_list = cell2mat(values(fold_accuracies));
times_list = cell2mat(values(fold_classification_times));

mean_accuracy = mean(accuracies_list);
std_accuracy = std(accuracies_list);
mean_time = mean(times_list);
std_time = std(times_list);

fprintf('\nAverage LOSO CV Accuracy: %.4f (+/- %.4f)\n', mean_accuracy, std_accuracy);
fprintf('Average classification time per sample: %.6f (+/- %.6f) seconds\n', mean_time, std_time);


conf_matrix = confusionmat(all_y_test, all_y_pred);


timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
accuracy_str = strrep(sprintf('acc_%.4f', mean_accuracy), '.', '_');
results_folder_name = sprintf('%s_%s_LOSO_LDMLT', timestamp, accuracy_str);
RESULTS_DIR = fullfile(RESULTS_DIR_BASE, results_folder_name);

if ~exist(RESULTS_DIR, 'dir')
    mkdir(RESULTS_DIR);
end
fprintf('\nResults will be saved in: %s\n', RESULTS_DIR);

fid = fopen(fullfile(RESULTS_DIR, 'detailed_accuracies.txt'), 'w');
fprintf(fid, 'Average LOSO CV Accuracy: %.4f (+/- %.4f)\n\n', mean_accuracy, std_accuracy);
fprintf(fid, 'Average classification time per sample: %.6f (+/- %.6f) seconds\n\n', mean_time, std_time);
fprintf(fid, 'Accuracy for each subject left out:\n');
fold_keys = keys(fold_accuracies);
for i = 1:length(fold_keys)
    fprintf(fid, '  - %s: %.4f\n', fold_keys{i}, fold_accuracies(fold_keys{i}));
end
fprintf(fid, '\nClassification time per sample for each subject left out:\n');
for i = 1:length(fold_keys)
    fprintf(fid, '  - %s: %.6f s\n', fold_keys{i}, fold_classification_times(fold_keys{i}));
end
fclose(fid);

fid = fopen(fullfile(RESULTS_DIR, 'classification_report.txt'), 'w');
fprintf(fid, 'Average LOSO CV Accuracy: %.4f (%.2f%%)\n\n', mean_accuracy, mean_accuracy * 100);
fprintf(fid, 'Average classification time per sample: %.6f seconds\n\n', mean_time);
fprintf(fid, 'Overall Classification Report (from all folds):\n');
fprintf(fid, '%12s %10s %10s %10s %10s\n', '', 'precision', 'recall', 'f1-score', 'support');

num_classes = length(unique_stroke_labels);
for i = 1:num_classes
    tp = sum(all_y_pred == i & all_y_test == i);
    fp = sum(all_y_pred == i & all_y_test ~= i);
    fn = sum(all_y_pred ~= i & all_y_test == i);
    
    precision = tp / (tp + fp);
    if isnan(precision), precision = 0; end
    
    recall = tp / (tp + fn);
    if isnan(recall), recall = 0; end
    
    f1_score = 2 * (precision * recall) / (precision + recall);
    if isnan(f1_score), f1_score = 0; end
    
    support = sum(all_y_test == i);
    
    fprintf(fid, '%12s %10.2f %10.2f %10.2f %10d\n', unique_stroke_labels{i}, precision, recall, f1_score, support);
end
fclose(fid);
disp('Classification report saved.');

fig = figure('Visible', 'off');
cm_chart = confusionchart(conf_matrix, unique_stroke_labels);
cm_chart.Title = sprintf('Aggregated Confusion Matrix (LOSO) - LDMLT+kNN\nAverage Accuracy: %.2f%%', mean_accuracy * 100);
saveas(fig, fullfile(RESULTS_DIR, 'confusion_matrix_loso.png'));
disp('LOSO confusion matrix plot saved.');

close(fig);
figure;
confusionchart(conf_matrix, unique_stroke_labels);
title(sprintf('Aggregated Confusion Matrix (LOSO) - LDMLT+kNN\nAverage Accuracy: %.2f%%', mean_accuracy * 100));

disp('>>> FINISHED <<<');
