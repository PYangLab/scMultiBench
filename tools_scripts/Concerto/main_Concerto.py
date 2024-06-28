import os
import sys
import bgi
import h5py
import time
import random
import argparse
import numpy as np
import scanpy as sc
import pandas as pd
from concerto_function5_3 import *
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, silhouette_samples

random.seed(1)
parser = argparse.ArgumentParser("Concerto")
parser.add_argument('--path1', metavar='DIR', nargs='+', default=[], help='path to train data1')
parser.add_argument('--path2', metavar='DIR', nargs='+', default=[], help='path to train data2')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
args = parser.parse_args()
begin_time = time.time()

def concerto_train_multimodal(RNA_tf_path: str, Protein_tf_path: str, weight_path: str, super_parameters=None):
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    # super_parameters is desined for multi_embedding_attention_transfer
    if super_parameters is None:
        super_parameters = {'batch_size': 64, 'epoch_pretrain': 3, 'lr': 1e-4,'drop_rate': 0.1}
    # dirname = os.getcwd()
    # f = np.load(ref_tf_path + './vocab_size.npz')
    f = np.load(os.path.join(RNA_tf_path, 'vocab_size.npz'))
    vocab_size_RNA = int(f['vocab size'])
    f = np.load(os.path.join(Protein_tf_path, 'vocab_size.npz'))
    vocab_size_Protein = int(f['vocab size'])
    encode_network = multi_embedding_attention_transfer(multi_max_features=[vocab_size_RNA,vocab_size_Protein],
                                                        mult_feature_names=['RNA','Protein'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)

    decode_network = multi_embedding_attention_transfer(multi_max_features=[vocab_size_RNA,vocab_size_Protein],
                                                        mult_feature_names=['RNA','Protein'],
                                                        embedding_dims=128,
                                                        include_attention=False,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)

    # tf_list_1 = os.listdir(os.path.join(ref_tf_path))
    tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]
    train_source_list_RNA = []
    train_source_list_Protein = []
    for i in tf_list_1:
        train_source_list_RNA.append(os.path.join(RNA_tf_path, i))
        train_source_list_Protein.append(os.path.join(Protein_tf_path, i))


    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_cls_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_cls_accuracy')
    test_cls_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_cls_accuracy')
    total_update_steps = 300 * super_parameters['epoch_pretrain']
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(super_parameters['lr'], total_update_steps,
                                                                super_parameters['lr'] * 1e-2, power=1)

    ########################################################
    # The only difference between this part and the original one is the following "Adam",
    opt_simclr = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    #opt_simclr = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)
    ########################################################

    for epoch in range(super_parameters['epoch_pretrain']):
        for RNA_file, Protein_file in zip(train_source_list_RNA, train_source_list_Protein):
            train_db_RNA = create_classifier_dataset_multi([RNA_file],
                                                           batch_size=super_parameters['batch_size'],
                                                           is_training=False,
                                                           data_augment=False,
                                                           shuffle_size=10000,
                                                           )
            train_db_Protein = create_classifier_dataset_multi([Protein_file],
                                                               batch_size=super_parameters['batch_size'],
                                                               is_training=False,
                                                               data_augment=False,
                                                               shuffle_size=10000,
                                                               )
            train_loss.reset_states()
            train_cls_accuracy.reset_states()
            test_cls_accuracy.reset_states()
            step = 0
            for (source_features_RNA, source_values_RNA,
                 source_batch_RNA, source_id_RNA), \
                (source_features_protein, source_values_protein,
                 source_batch_Protein, source_id_Protein) \
                    in (zip(train_db_RNA, train_db_Protein)):
                step += 1

                with tf.GradientTape() as tape:
                    z1 = encode_network([[source_features_RNA, source_features_protein],
                                         [source_values_RNA, source_values_protein]], training=True)
                    z2 = decode_network([source_values_RNA, source_values_protein], training=True)
                    ssl_loss = simclr_loss(z1, z2, temperature=0.1)
                    loss = ssl_loss
                    train_loss(loss)

                variables = [encode_network.trainable_variables,
                             decode_network.trainable_variables,
                             ]
                grads = tape.gradient(loss, variables)
                for grad, var in zip(grads, variables):
                    opt_simclr.apply_gradients(zip(grad, var))

                if step > 0 and step % 5 == 0:
                    template = 'Epoch {}, step {}, simclr loss: {:0.4f}.'
                    print(template.format(epoch + 1,
                                          str(step),
                                          train_loss.result()))
        encode_network.save_weights(
            os.path.join(weight_path, 'weight_encoder_epoch{}.h5'.format(str(epoch + 1))))
        decode_network.save_weights(
            os.path.join(weight_path, 'weight_decoder_epoch{}.h5'.format(str(epoch + 1))))

    return print('finished')

# The original one would report error if the type of input is byte.
def _bytes_feature(value):
    if isinstance(value, str):
        value = value.encode()
    elif hasattr(value, 'numpy'):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
import concerto_function5_3

concerto_function5_3._bytes_feature = _bytes_feature

def load_data_with_batch_and_label(adt_path, rna_path,  batch_name):
    with h5py.File(adt_path, 'r') as f:
        data_adt = np.array(f['matrix/data']).T
        barcodes_adt = np.array(f['matrix/barcodes'])
        features_adt = np.array(f['matrix/features'])

    with h5py.File(rna_path, 'r') as f:
        data_rna = np.array(f['matrix/data']).T
        barcodes_rna = np.array(f['matrix/barcodes'])
        features_rna = np.array(f['matrix/features'])

    adata_Protein = sc.AnnData(X=data_adt, obs=pd.DataFrame(index=barcodes_adt), var=pd.DataFrame(index=features_adt))
    adata_RNA = sc.AnnData(X=data_rna, obs=pd.DataFrame(index=barcodes_rna), var=pd.DataFrame(index=features_rna))
    adata_Protein.obs['batch'] = batch_name
    adata_RNA.obs['batch'] = batch_name
    return adata_Protein, adata_RNA

def get_adata_mod(adt_files, rna_files,  batch_name):
    adt_data_list = []
    rna_data_list = []

    for adt, rna, batch in zip(adt_files, rna_files, batch_name):
        adt_data, rna_data = load_data_with_batch_and_label(adt, rna, batch)
        adt_data_list.append(adt_data)
        rna_data_list.append(rna_data)

    standard_var_names = rna_data_list[0].var_names

    for adata in rna_data_list:
        adata.var_names = standard_var_names

    adata_Protein_combined = sc.concat(adt_data_list, axis=0, join='outer')
    adata_RNA_combined = sc.concat(rna_data_list, axis=0, join='outer')
    return adata_Protein_combined, adata_RNA_combined


def run_Concerto(adt_files, rna_files, batch_name):
  adata_Protein_combined, adata_RNA_combined = get_adata_mod(adt_files, rna_files, batch_name)
  adata_RNA_combined.X = np.nan_to_num(adata_RNA_combined.X, nan=0.0)
  #preprocessing the data
  adata_Protein_combined = preprocessing_rna(adata_Protein_combined,min_features = 0,is_hvg=False,batch_key='batch')
  adata_RNA_combined = preprocessing_rna(adata_RNA_combined,min_features = 0,is_hvg=False,batch_key='batch')
  
  save_path = './'
  if not os.path.exists(save_path):
      os.makedirs(save_path)
  adata_RNA_combined.write_h5ad(save_path + 'adata_RNA.h5ad')
  adata_Protein_combined.write_h5ad(save_path + 'adata_Protein.h5ad')

  #make TF-record
  RNA_tf_path = concerto_make_tfrecord(adata_RNA_combined,tf_path = save_path + 'tfrecord/RNA_tf/',batch_col_name= 'batch')
  Protein_tf_path = concerto_make_tfrecord(adata_Protein_combined,tf_path = save_path + 'tfrecord/Protein_tf/',batch_col_name= 'batch')
  #train(multimodal integration)
  save_path = './'
  weight_path = save_path + 'weight/'
  RNA_tf_path = save_path + 'tfrecord/RNA_tf/'
  Protein_tf_path = save_path + 'tfrecord/Protein_tf/'

  concerto_train_multimodal(RNA_tf_path,Protein_tf_path,weight_path,super_parameters = {'batch_size': 64, 'epoch_pretrain': 5, 'lr': 1e-4,'drop_rate': 0.1})  #5
  save_path = './'
  weight_path = save_path + 'weight/'
  RNA_tf_path = save_path + 'tfrecord/RNA_tf/'
  Protein_tf_path = save_path + 'tfrecord/Protein_tf/'
  saved_weight_path = None #'./weight/weight_encoder_epoch3.h5'# You can choose a trained weight or use None to default to the weight of the last epoch.
  embedding,batch,RNA_id,attention_weight =  concerto_test_multimodal(weight_path,RNA_tf_path,Protein_tf_path,n_cells_for_sample = None, super_parameters={'batch_size': 128, 'epoch_pretrain': 1, 'lr': 1e-4,'drop_rate': 0.1},saved_weight_path = saved_weight_path)

  return embedding

# run methods
adt_files = args.path2
rna_files = args.path1
batch_name = ["batch{}".format(i) for i in range(1, len(args.path1)+1)]
result = run_Concerto(adt_files, rna_files, batch_name)
end_time = time.time()
all_time = end_time - begin_time

# save results
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("create path")
else:
    print("the path exits")
file = h5py.File(args.save_path+"/embedding.h5", 'w')
file.create_dataset('data', data=result)
file.close()
np.savetxt(args.save_path+"/time.csv", [all_time], delimiter=",")
