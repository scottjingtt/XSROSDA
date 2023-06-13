import numpy as np
import os

dataset = 'D2LAD'

if dataset == 'D2AwA':
    src = ['AwA2_10', 'Painting_10', 'Real_10']
    tgt = ['AwA2_17', 'Painting_17', 'Real_17']
elif dataset == 'D2LAD':
    src = ['LAD_40', 'Painting_40', 'Real_40']
    tgt = ['LAD_56', 'Painting_56', 'Real_56']
else:
    raise Exception("dataset wrong!")

# 1. OSDA
print("OSDA")

base_results = []
prot_results = []
for i in range(3):
    for j in range(3):
        s = src[i]
        t = tgt[j]
        if s.split('_')[0] != t.split('_')[0]:

            checkpoint_path = os.path.join('./checkpoints', s+'2'+t)
            base_info = np.load(checkpoint_path + '/base_info.npy', allow_pickle=True).item()
            prot_info = np.load(checkpoint_path + '/prot_info.npy', allow_pickle=True).item()
            # init_info = np.load(checkpoint_path + 'init_info.npy', allow_pickle=True).item()

            print("Dataset: ", dataset, " Task: ", s, ' --> ', t)

            # 1.1 baseModel
            best_idx = np.argmax(base_info['osda']['OS'])
            print("baseModel: OS*={:.4}, OS^={:.4}, OS={:.4}, H={:.4}".format(
                base_info['osda']['OS*'][best_idx], base_info['osda']['OS^'][best_idx],
                base_info['osda']['OS'][best_idx], base_info['osda']['H'][best_idx]
            ))

            base_results.extend([str(round(r*100, 1)) for r in [base_info['osda']['OS*'][best_idx], base_info['osda']['OS^'][best_idx],
                base_info['osda']['OS'][best_idx], base_info['osda']['H'][best_idx]]])

            # 1.2 protoModel
            best_idx = np.argmax(prot_info['osda']['OS'])
            print("protoModel: OS*={:.4}, OS^={:.4}, OS={:.4}, H={:.4}".format(
                prot_info['osda']['OS*'][best_idx], prot_info['osda']['OS^'][best_idx],
                prot_info['osda']['OS'][best_idx], prot_info['osda']['H'][best_idx]
            ))
            prot_results.extend([str(round(r * 100, 1)) for r in [prot_info['osda']['OS*'][best_idx], prot_info['osda']['OS^'][best_idx],
                prot_info['osda']['OS'][best_idx], prot_info['osda']['H'][best_idx]]])

            print('--------------------------------')

print("baseModel: ", ' & '.join(base_results))

print("protoModel: ", ' & '.join(prot_results))
print("==========================================")
# 2. SR-OSDA

print("SR-OSDA")
base_results = []
prot_results = []

for i in range(3):
    for j in range(3):
        s = src[i]
        t = tgt[j]
        if s.split('_')[0] != t.split('_')[0]:
            checkpoint_path = os.path.join('./checkpoints', s+'2'+t)
            base_info = np.load(checkpoint_path + '/base_info.npy', allow_pickle=True).item()
            prot_info = np.load(checkpoint_path + '/prot_info.npy', allow_pickle=True).item()
            # init_info = np.load(checkpoint_path + 'init_info.npy', allow_pickle=True).item()

            print("Dataset: ", dataset, " Task: ", s, ' --> ', t)

            # 2.1 baseModel
            best_idx = np.argmax(base_info['srosda']['C_dis']['H'])
            print("baseModel: S={:.4}, U={:.4}, H={:.4}".format(
                base_info['srosda']['C_dis']['S'][best_idx], base_info['srosda']['C_dis']['U'][best_idx],
                base_info['srosda']['C_dis']['H'][best_idx]
            ))
            base_results.extend([str(round(r*100, 2)) for r in [base_info['srosda']['C_dis']['S'][best_idx], base_info['srosda']['C_dis']['U'][best_idx],
                base_info['srosda']['C_dis']['H'][best_idx]]])


            # 2.2 protoModel
            best_idx = np.argmax(prot_info['srosda']['C_dis']['H'])
            print("protoModel: S={:.4}, U={:.4}, H={:.4}".format(
                prot_info['srosda']['C_dis']['S'][best_idx], prot_info['srosda']['C_dis']['U'][best_idx],
                prot_info['srosda']['C_dis']['H'][best_idx]
            ))

            prot_results.extend([str(round(r * 100, 2)) for r in [prot_info['srosda']['C_dis']['S'][best_idx],
                                                                  prot_info['srosda']['C_dis']['U'][best_idx],
                                                                  prot_info['srosda']['C_dis']['H'][best_idx]]])


            print('-------------------------')

print("baseModel: ", ' & '.join(base_results))

print("protoModel: ", ' & '.join(prot_results))