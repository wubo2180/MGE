import pickle
import os
import argparse

import numpy as np


def pass_filter(filt, name):
    for f in filt:
        if f not in name:
            return False
    return True


def analysis(args):
    filt = args.filter.split(',')
    print('filter', filt)

    # construct result dict
    if args.mode not in ['bio', 'chem','dblp']:
        raise NotImplementedError('')
    parent_result_path = f'{args.mode}_result'
    print('parent_result_path',parent_result_path)
    result_dict = {}  
    filtered_seed_result_names = set()
    for seed_result_path in ["finetune_seed" + str(i) for i in range(5)]:
        full_result_path = os.path.join(parent_result_path, seed_result_path)

        if not os.path.exists(full_result_path):
            print(f'ommitting path {seed_result_path}')
            continue
        result_dict[seed_result_path] = {}
        # seed_result_names = os.listdir(full_result_path)
        seed_result_names = []
        print('full_result_path',full_result_path)
        for f in os.listdir(full_result_path):
            # if pass_filter(filt, f):
            seed_result_names.append(f)
        filtered_seed_result_names = filtered_seed_result_names.union(set(seed_result_names)) #[n for n in seed_result_names if split in n]
        #filtered_seed_result_names = ["speciessplit_cbow_l1-1_center0_epoch100", "speciessplit_gae_epoch100", "speciessplit_supervised_drop0.2_cbow_l1-1_center0_epoch100_epoch100"]
        print('seed_result_names',seed_result_names)
        for name in seed_result_names:
            #print(name)
            with open(os.path.join(parent_result_path, seed_result_path, name), "rb") as f:
                # print('os.path.join(parent_result_path, seed_result_path, name)',)
                result = pickle.load(f)
                result['train'] = result['train']
                result['val'] = result['val']
                # result['test_easy'] = result['test_easy']
                result['test_hard'] = result['test_hard']
            result_dict[seed_result_path][name] = result
    # print('result_dict',result_dict)
    #top_k = 40
    best_result_dict = {}  # dict[SEED#][experiment][test_easy/test_hard] = np.array, dim top_k classes
    for seed in result_dict:
        #print(seed)
        best_result_dict[seed] = {}
        for experiment in result_dict[seed]:
            #print(experiment)
            best_result_dict[seed][experiment] = {}
            #val = result_dict[seed][experiment]["val"][:, :top_k]  # look at the top k classes
            val = result_dict[seed][experiment]["val"]
            val_ave = np.average(val, axis=1)
            best_epoch = np.argmax(val_ave)
            
            # test_easy = result_dict[seed][experiment]["test_easy"]
            # test_easy_best = test_easy[best_epoch]
            
            #test_hard = result_dict[seed][experiment]["test_hard"][:, :top_k]
            test_hard = result_dict[seed][experiment]["test_hard"]
            test_hard_best = test_hard[best_epoch]
            
            # best_result_dict[seed][experiment]["test_easy"] = test_easy_best
            best_result_dict[seed][experiment]["test_hard"] = test_hard_best

    # average across the top k tasks and then average across all the seeds
    mean_result_dict = {}  # dict[experiment][test_easy/test_hard] = float
    std_result_dict = {}  # dict[experiment][test_easy/test_hard] = float
    print('filtered_seed_result_names',filtered_seed_result_names)
    for experiment in filtered_seed_result_names:
        print('experiment',experiment)
        mean_result_dict[experiment] = {}
        std_result_dict[experiment] = {}
        # test_easy_list = []
        test_hard_list = []
        for seed in best_result_dict:
            if experiment in best_result_dict[seed]:
                print(seed)
                # test_easy_list.append(best_result_dict[seed][experiment]['test_easy'])
                test_hard_list.append(best_result_dict[seed][experiment]['test_hard'])
        # mean_result_dict[experiment]['test_easy'] = np.array(test_easy_list).mean(axis=1).mean()
        mean_result_dict[experiment]['test_hard'] = np.array(test_hard_list).mean(axis=1).mean()
        # std_result_dict[experiment]['test_easy'] = np.array(test_easy_list).mean(axis=1).std()
        std_result_dict[experiment]['test_hard'] = np.array(test_hard_list).mean(axis=1).std()
    print('test_hard_list',len(test_hard_list[0]))
    print('mean_result_dict',mean_result_dict)
    # results test hard
    sorted_test_hard = sorted(mean_result_dict.items(), key=lambda kv: kv[1]['test_hard'], reverse=True)
    print('sorted_test_hard',sorted_test_hard)
    for k, _ in sorted_test_hard:
        # print(k)
        print('{:.2f} ± {:.2f}'.format(mean_result_dict[k]['test_hard']*100, std_result_dict[k]['test_hard']*100))
        
# def analysis(args):
#     filt = args.filter.split(',')
#     print('filter', filt)

#     # construct result dict
#     if args.mode not in ['bio', 'chem','dblp']:
#         raise NotImplementedError('')
#     parent_result_path = f'{args.mode}_result'
#     print('parent_result_path',parent_result_path)
#     result_dict = {}  
#     filtered_seed_result_names = set()
#     for seed_result_path in ["finetune_seed" + str(i) for i in range(1)]:
#         full_result_path = os.path.join(parent_result_path, seed_result_path)

#         if not os.path.exists(full_result_path):
#             print(f'ommitting path {seed_result_path}')
#             continue
#         result_dict[seed_result_path] = {}
#         # seed_result_names = os.listdir(full_result_path)
#         seed_result_names = []
#         print('full_result_path',full_result_path)
#         for f in os.listdir(full_result_path):
#             # if pass_filter(filt, f):
#             seed_result_names.append(f)
#         filtered_seed_result_names = filtered_seed_result_names.union(set(seed_result_names)) #[n for n in seed_result_names if split in n]
#         #filtered_seed_result_names = ["speciessplit_cbow_l1-1_center0_epoch100", "speciessplit_gae_epoch100", "speciessplit_supervised_drop0.2_cbow_l1-1_center0_epoch100_epoch100"]
#         print('seed_result_names',seed_result_names)
#         for name in seed_result_names:
#             #print(name)
#             with open(os.path.join(parent_result_path, seed_result_path, name), "rb") as f:
#                 # print('os.path.join(parent_result_path, seed_result_path, name)',)
#                 result = pickle.load(f)
#                 result['train'] = result['train']
#                 result['val'] = result['val']
#                 # result['test_easy'] = result['test_easy']
#                 result['test'] = result['test']
#             result_dict[seed_result_path][name] = result
#     # print('result_dict',result_dict)
#     #top_k = 40
#     best_result_dict = {}  # dict[SEED#][experiment][test_easy/test] = np.array, dim top_k classes
#     for seed in result_dict:
#         #print(seed)
#         best_result_dict[seed] = {}
#         for experiment in result_dict[seed]:
#             #print(experiment)
#             best_result_dict[seed][experiment] = {}
#             #val = result_dict[seed][experiment]["val"][:, :top_k]  # look at the top k classes
#             val = result_dict[seed][experiment]["val"]
#             val_ave = np.average(val, axis=1)
#             best_epoch = np.argmax(val_ave)
            
#             # test_easy = result_dict[seed][experiment]["test_easy"]
#             # test_easy_best = test_easy[best_epoch]
            
#             #test = result_dict[seed][experiment]["test"][:, :top_k]
#             test = result_dict[seed][experiment]["test"]
#             test_best = test[best_epoch]
            
#             # best_result_dict[seed][experiment]["test_easy"] = test_easy_best
#             best_result_dict[seed][experiment]["test"] = test_best

#     # average across the top k tasks and then average across all the seeds
#     mean_result_dict = {}  # dict[experiment][test_easy/test] = float
#     std_result_dict = {}  # dict[experiment][test_easy/test] = float
#     print('filtered_seed_result_names',filtered_seed_result_names)
#     for experiment in filtered_seed_result_names:
#         print('experiment',experiment)
#         mean_result_dict[experiment] = {}
#         std_result_dict[experiment] = {}
#         # test_easy_list = []
#         test_list = []
#         for seed in best_result_dict:
#             if experiment in best_result_dict[seed]:
#                 print(seed)
#                 # test_easy_list.append(best_result_dict[seed][experiment]['test_easy'])
#                 test_list.append(best_result_dict[seed][experiment]['test'])
#         # mean_result_dict[experiment]['test_easy'] = np.array(test_easy_list).mean(axis=1).mean()
#         mean_result_dict[experiment]['test'] = np.array(test_list).mean(axis=1).mean()
#         # std_result_dict[experiment]['test_easy'] = np.array(test_easy_list).mean(axis=1).std()
#         std_result_dict[experiment]['test'] = np.array(test_list).mean(axis=1).std()
#     print('test_list',len(test_list[0]))
#     print('mean_result_dict',mean_result_dict)
#     # results test hard
#     sorted_test = sorted(mean_result_dict.items(), key=lambda kv: kv[1]['test'], reverse=True)
#     print('sorted_test',sorted_test)
#     for k, _ in sorted_test:
#         # print(k)
#         print('{:.2f} ± {:.2f}'.format(mean_result_dict[k]['test']*100, std_result_dict[k]['test']*100))
#         print("")
def analysis_dblp(args):
    # construct result dict
    if args.mode not in ['bio', 'chem','dblp']:
        raise NotImplementedError('')
    parent_result_path = f'{args.mode}_result'
    print('parent_result_path',parent_result_path)
    result_dict = {}  
    filtered_seed_result_names = set()
    for seed_result_path in ["finetune_seed" + str(i) for i in range(1)]:
        full_result_path = os.path.join(parent_result_path, seed_result_path)

        if not os.path.exists(full_result_path):
            print(f'ommitting path {seed_result_path}')
            continue
        result_dict[seed_result_path] = {}
        # seed_result_names = os.listdir(full_result_path)
        seed_result_names = []
        print('full_result_path',full_result_path)
        for f in os.listdir(full_result_path):
            # if pass_filter(filt, f):
            seed_result_names.append(f)
        filtered_seed_result_names = filtered_seed_result_names.union(set(seed_result_names)) #[n for n in seed_result_names if split in n]
        #filtered_seed_result_names = ["speciessplit_cbow_l1-1_center0_epoch100", "speciessplit_gae_epoch100", "speciessplit_supervised_drop0.2_cbow_l1-1_center0_epoch100_epoch100"]
        print('seed_result_names',seed_result_names)
        for name in seed_result_names:
            #print(name)
            with open(os.path.join(parent_result_path, seed_result_path, name), "rb") as f:
                # print('os.path.join(parent_result_path, seed_result_path, name)',)
                result = pickle.load(f)
                result['train'] = result['train']
                result['val'] = result['val']
                # result['test_easy'] = result['test_easy']
                result['test'] = result['test']
            result_dict[seed_result_path][name] = result

    # top_k = 40
    best_result_dict = {}  # dict[SEED#][experiment][test_easy/test_hard] = np.array, dim top_k classes
    for seed in result_dict:
        # print(seed)
        best_result_dict[seed] = {}
        for experiment in result_dict[seed]:
            best_result_dict[seed][experiment] = {}
            val = result_dict[seed][experiment]["val"]
            best_epoch = np.argmax(val)

            test_easy = result_dict[seed][experiment]["test"]
            test_easy_best = test_easy[best_epoch]

            best_result_dict[seed][experiment]["test"] = test_easy_best

    # average across the top k tasks and then average across all the seeds
    mean_result_dict = {}  # dict[experiment][test_easy/test_hard] = float
    std_result_dict = {}  # dict[experiment][test_easy/test_hard] = float
    for experiment in filtered_seed_result_names:
        # print(experiment)
        mean_result_dict[experiment] = {}
        std_result_dict[experiment] = {}
        test_easy_list = []
        for seed in best_result_dict:
            test_easy_list.append(best_result_dict[seed][experiment]['test'])
        mean_result_dict[experiment]['test'] = np.array(test_easy_list).mean()
        std_result_dict[experiment]['test'] = np.array(test_easy_list).std()

    # results test hard
    sorted_test_hard = sorted(mean_result_dict.items(), key=lambda kv: kv[1]['test'], reverse=True)
    for k, _ in sorted_test_hard:
        print(k)
        print('{} +- {}'.format(mean_result_dict[k]['test'] * 100, std_result_dict[k]['test'] * 100))
        print("")
def analysis_chem(args):
    if args.mode not in ['bio', 'chem','dblp']:
        raise NotImplementedError('')
    parent_result_path = f'{args.mode}_result/'+args.dataset
    print('parent_result_path',parent_result_path)
    result_dict = {}  
    filtered_seed_result_names = set()
    for seed_result_path in ["finetune_seed" + str(i) for i in range(1)]:
        full_result_path = os.path.join(parent_result_path, seed_result_path)

        if not os.path.exists(full_result_path):
            print(f'ommitting path {seed_result_path}')
            continue
        result_dict[seed_result_path] = {}
        # seed_result_names = os.listdir(full_result_path)
        seed_result_names = []
        print('full_result_path',full_result_path)
        for f in os.listdir(full_result_path):
            # if pass_filter(filt, f):
            seed_result_names.append(f)
        filtered_seed_result_names = filtered_seed_result_names.union(set(seed_result_names)) #[n for n in seed_result_names if split in n]
        #filtered_seed_result_names = ["speciessplit_cbow_l1-1_center0_epoch100", "speciessplit_gae_epoch100", "speciessplit_supervised_drop0.2_cbow_l1-1_center0_epoch100_epoch100"]
        print('seed_result_names',seed_result_names)
        for name in seed_result_names:
            #print(name)
            with open(os.path.join(parent_result_path, seed_result_path, name), "rb") as f:
                # print('os.path.join(parent_result_path, seed_result_path, name)',)
                result = pickle.load(f)
                result['train'] = result['train']
                result['val'] = result['val']
                # result['test_easy'] = result['test_easy']
                result['test'] = result['test']
            result_dict[seed_result_path][name] = result

    # top_k = 40
    best_result_dict = {}  # dict[SEED#][experiment][test_easy/test_hard] = np.array, dim top_k classes
    for seed in result_dict:
        # print(seed)
        best_result_dict[seed] = {}
        for experiment in result_dict[seed]:
            best_result_dict[seed][experiment] = {}
            val = result_dict[seed][experiment]["val"]
            best_epoch = np.argmax(val)

            test_easy = result_dict[seed][experiment]["test"]
            test_easy_best = test_easy[best_epoch]

            best_result_dict[seed][experiment]["test"] = test_easy_best

    # average across the top k tasks and then average across all the seeds
    mean_result_dict = {}  # dict[experiment][test_easy/test_hard] = float
    std_result_dict = {}  # dict[experiment][test_easy/test_hard] = float
    for experiment in filtered_seed_result_names:
        # print(experiment)
        mean_result_dict[experiment] = {}
        std_result_dict[experiment] = {}
        test_easy_list = []
        for seed in best_result_dict:
            test_easy_list.append(best_result_dict[seed][experiment]['test'])
        mean_result_dict[experiment]['test'] = np.array(test_easy_list).mean()
        std_result_dict[experiment]['test'] = np.array(test_easy_list).std()

    # results test hard
    sorted_test_hard = sorted(mean_result_dict.items(), key=lambda kv: kv[1]['test'], reverse=True)
    for k, _ in sorted_test_hard:
        print(k)
        print('{} +- {}'.format(mean_result_dict[k]['test'] * 100, std_result_dict[k]['test'] * 100))
        print("")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='chem', help='bio or chem or dblp.')
    parser.add_argument('--filter', type=str, default='show results contain the input filter string.')
    parser.add_argument('--dataset', type=str, default='bbbp',help='chem downstream dataset.')
    args = parser.parse_args()
    print(args, flush=True)
    if args.mode=='bio':
        analysis(args)
    elif args.mode=='chem':
        analysis_chem(args)
    elif args.mode=='dblp':
        analysis_dblp(args)
