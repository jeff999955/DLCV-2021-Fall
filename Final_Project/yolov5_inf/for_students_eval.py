import os
import csv
import argparse
import numpy as np
from sklearn.cluster import DBSCAN

def cal_TPFNFP(preds, gts, r=32):
	# preds = [x1, y1, x2, y2, x3, y3, ...], gts = [x1, y1, x2, y2, ...]
	# Manhattan distance = |x1 - x2| + |y1 - y2|
	assert len(preds) % 2 == 0
	assert len(gts) % 2 == 0

	gt_selected = [False for _ in range(len(gts) // 2)]
	neg_preds = []

	for i in range(0, len(preds), 2):
		flag = False 				# False indicates current node doesn't fall in any circle
		for j in range(0, len(gts), 2):
			if (abs(int(preds[i]) - int(gts[j])) + abs(int(preds[i+1]) - int(gts[j+1]))) <= r:
				flag = True
				gt_selected[j // 2] = True
		if not flag:
			neg_preds.append([int(preds[i]), int(preds[i+1])])

	TP = sum(gt_selected)
	FN = len(gts) // 2 - TP
	FP = 0
	
	if len(neg_preds) > 0:
		neg_preds = np.array(neg_preds)		# neg_preds size should be (n, 2)
		clustering = DBSCAN(eps=32, metric='manhattan', min_samples=1).fit(neg_preds)
		FP = len(np.unique(clustering.labels_))

	return TP, FN, FP

def cal_mean_hit_rate(preds, gts, image_name):
	# preds and gts format example = [['12', '203', '294', '1024'], ['39', '95', '283', '94']]

	hit_rate_patient = []
	cur_patient = {'name':image_name[0][:20], 'TP':0, 'FP':0, 'FN':0}
	for i in range(len(preds)):
		if image_name[i][:20] == cur_patient['name']:
			TP, FN, FP = cal_TPFNFP(preds[i], gts[i])		# for each image of a patient
			cur_patient['TP'] += TP
			cur_patient['FN'] += FN
			cur_patient['FP'] += FP
		else:
			hit_rate = float(cur_patient['TP']) / (cur_patient['TP'] + cur_patient['FN'] + cur_patient['FP']) if (cur_patient['TP'] + cur_patient['FN'] + cur_patient['FP']) != 0 else -1
			hit_rate_patient.append(hit_rate)
			TP, FN, FP = cal_TPFNFP(preds[i], gts[i])
			cur_patient['name'] = image_name[i][:20]
			cur_patient['TP'] = TP
			cur_patient['FN'] = FN
			cur_patient['FP'] = FP
	hit_rate = float(cur_patient['TP']) / (cur_patient['TP'] + cur_patient['FN'] + cur_patient['FP']) if (cur_patient['TP'] + cur_patient['FN'] + cur_patient['FP']) != 0 else -1
	hit_rate_patient.append(hit_rate)
	hit_rate_patient = np.array(hit_rate_patient)
	assert len(hit_rate_patient) == 130, "Incorrect number of patients when calculating mean hit rate"

	s = sum(x for x in hit_rate_patient if x >= 0)
	total = sum(hit_rate_patient >= 0)

	return float(s) / total

def cal_F1(preds, gts, image_name):
	# preds and gts format example = [['12', '203', '294', '1024'], ['39', '95', '283', '94']]

	statistics = {'TP':0, 'FP':0, 'FN':0}
	for i in range(len(preds)):
		TP, FN, FP = cal_TPFNFP(preds[i], gts[i])		# for each image of a patient
		statistics['TP'] += TP
		statistics['FN'] += FN
		statistics['FP'] += FP
	F1_score = float(2 * statistics['TP']) / (2 * statistics['TP'] + statistics['FN'] + statistics['FP'])

	return statistics['TP'], statistics['FN'], statistics['FP'], F1_score

def cal_acc(preds, gts, image_name):
	correct = 0
	total = 0
	cur_patient = image_name[0][:20]
	cur_patient_gt = 1
	cur_patient_pred = 1
	for i in range(len(preds)):
		if image_name[i][:20] == cur_patient:
			cur_patient_gt *= int(gts[i])
			cur_patient_pred *= int(preds[i])
		else:
			correct += (abs(cur_patient_gt) == abs(cur_patient_pred))
			total += 1
			cur_patient = image_name[i][:20]
			cur_patient_gt = int(gts[i])
			cur_patient_pred = int(preds[i])
	correct += (abs(cur_patient_gt) == abs(cur_patient_pred))
	total += 1
	print(total)
	# assert total == 130, "Incorrect number of patients when calculating accuracy"	# 280 patients in validation data
	case_level_acc = float(correct) / total

	return case_level_acc
	
def evaluate(prediction_file, groundtruth_file):
	image_name = []
	pred_label = []
	gt_label = []
	pred_point = []
	gt_point = []

	# csv row[0]: image name, row[1]: label (0, 1, -1), row[2]: x1, row[3]: y1, row[4]: x2, row[5]: y2, ...
	with open(prediction_file, newline='') as csvfile:
		csv_reader = csv.reader(csvfile)
		next(csv_reader)
		for row in csv_reader:
			pred_label.append(int(row[1]))
			pred_point.append(row[2].strip().split(' ') if len(row[2]) != 0 else [])
			image_name.append(row[0])

	i = 0
	with open(groundtruth_file, newline='') as csvfile:
		csv_reader = csv.reader(csvfile)
		next(csv_reader)
		for row in csv_reader:
			gt_label.append(int(row[1]))
			gt_point.append(row[2].strip().split(' ') if len(row[2]) != 0 else [])
			assert image_name[i] == row[0]			# To ensure the order of image names in pred is the same as the order in gt
			i += 1

	case_level_acc = cal_acc(pred_label, gt_label, image_name)
	TP, FN, FP, F1_score = cal_F1(pred_point, gt_point, image_name)		# mean hit rate for every person
	print('TP = {}, FN = {}, FP = {}'.format(TP, FN, FP))

	return case_level_acc, F1_score

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--pred_file", type=str, required=True)
	parser.add_argument("--gt_file", type=str)
	args = parser.parse_args()

	case_level_acc, F1_score = evaluate(args.pred_file, args.gt_file)
	print('Case level acc = {:.4f} | F1_score = {:.4f}'.format(case_level_acc, F1_score))
