import os
import h5py
import torch
import random
import argparse
import numpy as np
from copy import deepcopy
from model_utils import Net
from eval_utils import infer_result, save_result
from data_utils import prepare_dataloader, partition_data, adjacency

# The scBridge script for diagonal integration requires RNA and ATAC data as input and the cell type labels, where ATAC needs to be transformed into gene activity score. The output is a joint embedding (dimensionality reduction).
# run commond for scBridge
# python main_scBridge.py --data_path="../../data/dataset_final/D27/" --source_data="rna.h5" --target_data="atac_gas.h5" --source_cty "rna_cty.csv" --target_cty "atac_cty.csv" --save_path="../../result/embedding/diagonal integration/D27/scBridge/"  --umap_plot

def main(args):
    (
        source_dataset,
        source_dataloader_train,
        source_dataloader_eval,
        target_dataset,
        target_dataloader_train,
        target_dataloader_eval,
        gene_num,
        type_num,
        label_map,
        source_adata,
        target_adata,
    ) = prepare_dataloader(args)

    source_dataloader_eval_all = deepcopy(source_dataloader_eval)
    target_dataloader_eval_all = deepcopy(target_dataloader_eval)
    if args.novel_type:
        target_adj = adjacency(target_dataset.tensors[0])
    else:
        target_adj = None

    source_label = source_dataset.tensors[1]
    count = torch.unique(source_label, return_counts=True, sorted=True)[1]
    ce_weight = 1.0 / count
    ce_weight = ce_weight / ce_weight.sum() * type_num
    ce_weight = ce_weight.cuda()

    print("======= Training Start =======")

    net = Net(gene_num, type_num, ce_weight, args).cuda()
    preds, prob_feat, prob_logit = net.run(
        source_dataloader_train,
        source_dataloader_eval,
        target_dataloader_train,
        target_dataloader_eval,
        target_adj,
        args,
    )

    for iter in range(args.max_iteration):
        (
            source_dataloader_train,
            source_dataloader_eval,
            target_dataloader_train,
            target_dataloader_eval,
            source_dataset,
            target_dataset,
        ) = partition_data(
            preds,
            prob_feat,
            prob_logit,
            source_dataset,
            target_dataset,
            args,
        )

        # Iteration convergence check
        if target_dataset.__len__() <= args.batch_size:
            break
        print("======= Iteration:", iter, "=======")

        source_label = source_dataset.tensors[1]
        count = torch.unique(source_label, return_counts=True, sorted=True)[1]
        ce_weight = 1.0 / count
        ce_weight = ce_weight / ce_weight.sum() * type_num
        ce_weight = ce_weight.cuda()

        net = Net(gene_num, type_num, ce_weight, args).cuda()
        preds, prob_feat, prob_logit = net.run(
            source_dataloader_train,
            source_dataloader_eval,
            target_dataloader_train,
            target_dataloader_eval,
            target_adj,
            args,
        )
    print("======= Training Done =======")

    features, predictions, reliabilities = infer_result(
        net, source_dataloader_eval_all, target_dataloader_eval_all, args
    )
    
    source_adata, target_adata = save_result(
        features,
        predictions,
        reliabilities,
        label_map,
        type_num,
        source_adata,
        target_adata,
        args,
    )
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print("create path")
    else:
        print("the path exits")
        
    embedding = np.concatenate([source_adata.obsm['Embedding'], target_adata.obsm['Embedding']],0)
    print(embedding.shape)

    file = h5py.File(args.save_path+"/embedding.h5", 'w')
    file.create_dataset('data', data=embedding)
    file.close()

    np.savetxt(args.save_path+"/Prediction.csv", target_adata.obs['Prediction'].values.tolist(), fmt='%s', delimiter=",")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data configs
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--source_data", type=str)
    parser.add_argument("--source_cty", type=str, default="cty.csv")
    parser.add_argument("--target_data", type=str)
    parser.add_argument("--target_cty", type=str, default="cty.csv")
    parser.add_argument("--source_preprocess", type=str, default="Standard")
    parser.add_argument("--target_preprocess", type=str, default="TFIDF")
    parser.add_argument("--save_path", type=str, default="./")
    # Model configs
    parser.add_argument("--reliability_threshold", default=0.95, type=float)
    parser.add_argument("--align_loss_epoch", default=1, type=float)
    parser.add_argument("--prototype_momentum", default=0.9, type=float)
    parser.add_argument("--early_stop_acc", default=0.99, type=float)
    parser.add_argument("--max_iteration", default=20, type=int)
    parser.add_argument("--novel_type", action="store_true")
    # Training configs
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--train_epoch", default=20, type=int)
    parser.add_argument("--learning_rate", default=5e-4, type=float)
    parser.add_argument("--random_seed", default=2023, type=int)
    # Evaluation configs
    parser.add_argument("--umap_plot", action="store_false")

    args = parser.parse_args()

    # Randomization
    torch.manual_seed(args.random_seed)
    torch.random.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    main(args)
