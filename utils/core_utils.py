from ast import Lambda
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import pdb
import os
from custom_optims.radam import RAdam

from models.model_DPMSurv import DPMSurv

from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, brier_score, integrated_brier_score, cumulative_dynamic_auc
from sksurv.util import Surv

from transformers import (
    get_constant_schedule_with_warmup, 
    get_linear_schedule_with_warmup, 
    get_cosine_schedule_with_warmup
)

import torch
from torch.nn.utils.rnn import pad_sequence

from utils.general_utils import _get_split_loader, _print_network, _save_splits
from utils.loss_func import NLLSurvLoss

import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_splits(datasets, cur, args):
    print('\nTraining Fold {}!'.format(cur))
    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    _save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))
    return train_split, val_split, test_split

def _init_loss_function(args):
    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'nll_surv':
        loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
    else:
        raise NotImplementedError
    print('Done!')
    return loss_fn

def _init_optim(args, model):
    print('\nInit optimizer ...', end=' ')
    if args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    elif args.opt == "adamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.reg)
    elif args.opt == "radam":
        optimizer = RAdam(model.parameters(), lr=args.lr, weight_decay=args.reg)
    elif args.opt == "lamb":
        optimizer = Lambda(model.parameters(), lr=args.lr, weight_decay=args.reg)
    else:
        raise NotImplementedError
    return optimizer

def _init_model(args):
    print('\nInit Model...', end=' ')
    if args.type_of_path == "xena":
        omics_input_dim = 1577
    elif args.type_of_path == "hallmarks":
        omics_input_dim = 4241
    elif args.type_of_path == "combine":
        omics_input_dim = 4999
    elif args.type_of_path == "multi":
        if args.study == "tcga_brca":
            omics_input_dim = 9947
        else:
            omics_input_dim = 14933
    else:
        omics_input_dim = 0
    
    if args.modality == "DPMSurv":
        model_dict = {
             "omic_input_dim" : omics_input_dim, "dropout": args.encoder_dropout,
             "mil_model_type": args.mil_model_type, "geno_mlp_type":args.geno_mlp_type,
             "memory_size": args.memory_size,"update_topk": getattr(args, 'update_topk', 0),
        }
        model = DPMSurv(**model_dict)
    else:
        raise NotImplementedError

    if torch.cuda.is_available():
        model = model.to(device)

    print('Done!')
    _print_network(args.results_dir, model)
    return model

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss

def _init_loaders(args, train_split, val_split, test_split):
    print('\nInit Loaders...', end=' ')
    if train_split:
        train_loader = _get_split_loader(args, train_split, training=True, testing=False, weighted=args.weighted_sample, batch_size=args.batch_size)
    else:
        train_loader = None
    if val_split:
        val_loader = _get_split_loader(args, val_split, testing=False, batch_size=1)
    else:
        val_loader = None
    if test_split:
        test_loader = _get_split_loader(args, test_split, testing=False, batch_size=1)
    else:
        test_loader = None
    print('Done!')
    return train_loader, val_loader, test_loader

def _extract_survival_metadata(train_loader, val_loader, test_loader):
    all_censorships = np.concatenate(
        [train_loader.dataset.metadata[train_loader.dataset.censorship_var].to_numpy(),
        val_loader.dataset.metadata[val_loader.dataset.censorship_var].to_numpy(),
        test_loader.dataset.metadata[test_loader.dataset.censorship_var].to_numpy()],
        axis=0)
    all_event_times = np.concatenate(
        [train_loader.dataset.metadata[train_loader.dataset.label_col].to_numpy(),
        val_loader.dataset.metadata[val_loader.dataset.label_col].to_numpy(),
        test_loader.dataset.metadata[test_loader.dataset.label_col].to_numpy()],
        axis=0)
    all_survival = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
    return all_survival

def _unpack_data(modality, device, data):
    if modality in ["DPMSurv"]:
        data_WSI = data[0].to(device)
        data_omics = data[1].to(device)
        if data[6][0,0] == 1:
            mask = None
        else:
            mask = data[6].to(device)
        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]
    else:
        raise ValueError('Unsupported modality:', modality)
    y_disc, event_time, censor = y_disc.to(device), event_time.to(device), censor.to(device)
    return data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask

def _process_data_and_forward(model, modality, device, data, args):
    data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask = _unpack_data(modality, device, data)
   
    if modality in ["DPMSurv"]:
        input_args = {"x_path": data_WSI.to(device)}
        input_args["x_omic"] = data_omics.to(device)
        input_args["label"] = y_disc
        input_args['censor'] = censor
        input_args['training'] = True

        if args.complete_rate == 1.0:
            input_args['input_modality'] = 'path_and_geno'
            out = model(**input_args)
        elif args.complete_rate > 0:
            modality_list = ['path_and_geno', 'path', 'geno']
            rate = args.complete_rate
            weight_list = [rate, (1-rate)/2, (1-rate)/2]
            input_modality = np.random.choice(modality_list, p=weight_list)
            input_args['input_modality'] = input_modality
            if input_modality == 'path_and_geno':
                pass
            elif input_modality == 'path':
                input_args["x_omic"] = None
            elif input_modality == 'geno':
                input_args["x_path"] = None
            out = model(**input_args)
        else:
            raise ValueError('Unsupported complete data rate:', args.complete_rate)
    else:
        raise NotImplementedError
        
    return out, y_disc, event_time, censor, clinical_data_list

def _calculate_risk(h):
    hazards = torch.sigmoid(h)
    survival = torch.cumprod(1 - hazards, dim=1)
    risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
    return risk, survival.detach().cpu().numpy()

def _update_arrays(all_risk_scores, all_censorships, all_event_times, all_clinical_data, event_time, censor, risk, clinical_data_list):
    all_risk_scores.append(risk)
    all_censorships.append(censor.detach().cpu().numpy())
    all_event_times.append(event_time.detach().cpu().numpy())
    all_clinical_data.append(clinical_data_list)
    return all_risk_scores, all_censorships, all_event_times, all_clinical_data

def _train_loop_survival(epoch, model, modality, loader, optimizer, scheduler, loss_fn, args):
    model.train()

    total_loss = 0.
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    all_clinical_data = []

    for batch_idx, data in enumerate(loader):
        
        optimizer.zero_grad()

        if modality in ["DPMSurv"]:
            out, y_disc, event_time, censor, clinical_data_list = _process_data_and_forward(model, modality, device, data, args)
            h, sim_loss, loss_align, loss_proto_align, loss_proto_triplet = out
            surv_loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)

            weight2 = args.sim_loss
            weight3 = args.align_loss
            weight4 = getattr(args, 'proto_align_loss',   0.1)
            weight5 = getattr(args, 'proto_triplet_loss', 0.1)

            if batch_idx == 0 and epoch % 5 == 0:
                surv_val         = surv_loss.item()
                sim_val          = sim_loss.item() if hasattr(sim_loss, 'item') else float(sim_loss)
                align_val        = loss_align.item() if hasattr(loss_align, 'item') else float(loss_align)
                proto_align_val  = loss_proto_align.item() if hasattr(loss_proto_align, 'item') else float(loss_proto_align)
                proto_triplet_val= loss_proto_triplet.item() if hasattr(loss_proto_triplet, 'item') else float(loss_proto_triplet)
                print(f"\n[DEBUG Plan-M epoch={epoch}]")
                print(f"  surv_loss          = {surv_val:.4f}")
                print(f"  sim_loss (raw)     = {sim_val:.4f}  × {weight2} = {weight2*sim_val:.4f}")
                print(f"  align_loss (raw)   = {align_val:.4f}  × {weight3} = {weight3*align_val:.4f}")
                print(f"  proto_align (raw)  = {proto_align_val:.4f}  × {weight4} = {weight4*proto_align_val:.4f}")
                print(f"  proto_triplet(raw) = {proto_triplet_val:.4f}  × {weight5} = {weight5*proto_triplet_val:.4f}")
                print(f"  total (this batch) = {surv_val + weight2*sim_val + weight3*align_val + weight4*proto_align_val + weight5*proto_triplet_val:.4f}\n")

            if args.use_align_loss:
                loss = (surv_loss
                        + weight2 * sim_loss
                        + weight3 * loss_align
                        + weight4 * loss_proto_align
                        + weight5 * loss_proto_triplet)
            else:
                loss = (surv_loss
                        + weight2 * sim_loss
                        + weight4 * loss_proto_align
                        + weight5 * loss_proto_triplet)

        else:
            raise NotImplementedError   
            
        loss_value = loss.item()
        loss = loss / y_disc.shape[0]

        risk, _ = _calculate_risk(h)

        all_risk_scores, all_censorships, all_event_times, all_clinical_data = _update_arrays(
            all_risk_scores, all_censorships, all_event_times, all_clinical_data,
            event_time, censor, risk, clinical_data_list)

        total_loss += loss_value 

        loss.backward()
        optimizer.step()
        scheduler.step()

    total_loss /= len(loader.dataset)

    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    print('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, total_loss, c_index))

    return c_index, total_loss

def _calculate_metrics(loader, dataset_factory, survival_train, all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores):
    data = loader.dataset.metadata["survival_months_dss"]
    bins_original = dataset_factory.bins
    which_times_to_eval_at = np.array([data.min() + 0.0001, bins_original[1], bins_original[2], data.max() - 0.0001])

    original_risk_scores = all_risk_scores
    all_risk_scores  = np.delete(all_risk_scores,  np.argwhere(np.isnan(original_risk_scores)))
    all_censorships  = np.delete(all_censorships,  np.argwhere(np.isnan(original_risk_scores)))
    all_event_times  = np.delete(all_event_times,  np.argwhere(np.isnan(original_risk_scores)))

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    c_index_ipcw, BS, IBS, iauc = 0., 0., 0., 0.

    try:
        survival_test = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
    except:
        print("Problem converting survival test datatype, so all metrics 0.")
        return c_index, c_index_ipcw, BS, IBS, iauc

    try:
        c_index_ipcw = concordance_index_ipcw(survival_train, survival_test, estimate=all_risk_scores)[0]
    except:
        print('An error occured while computing c-index ipcw')
        c_index_ipcw = 0.

    try:
        _, BS = brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores, times=which_times_to_eval_at)
    except:
        print('An error occured while computing BS')
        BS = 0.

    try:
        IBS = integrated_brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores, times=which_times_to_eval_at)
    except:
        print('An error occured while computing IBS')
        IBS = 0.

    try:
        _, iauc = cumulative_dynamic_auc(survival_train, survival_test, estimate=1-all_risk_by_bin_scores[:, 1:], times=which_times_to_eval_at[1:])
    except:
        print('An error occured while computing iauc')
        iauc = 0.

    return c_index, c_index_ipcw, BS, IBS, iauc

def _summary(dataset_factory, model, modality, loader, loss_fn, survival_train=None, input_modality='path_and_geno', args=None):
    model.eval()

    total_loss = 0.
    all_risk_scores = []
    all_risk_by_bin_scores = []
    all_censorships = []
    all_event_times = []
    all_clinical_data = []
    all_logits = []
    all_slide_ids = []

    slide_ids = loader.dataset.metadata['slide_id']
    count = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask = _unpack_data(modality, device, data)

            if modality in ["DPMSurv"]:
                input_args = {"x_path": data_WSI.to(device)}
                input_args["x_omic"] = data_omics.to(device)
                input_args["label"] = None
                input_args['censor'] = None
                input_args['training'] = False
                input_args['input_modality'] = input_modality
                input_args['return_feature'] = False
                h = model(**input_args)
            else:
                raise NotImplementedError
                    
            if len(h.shape) == 1:
                h = h.unsqueeze(0)

            loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
            loss_value = loss.item()
            loss = loss / y_disc.shape[0]

            risk, risk_by_bin = _calculate_risk(h)
            all_risk_by_bin_scores.append(risk_by_bin)
            all_risk_scores, all_censorships, all_event_times, clinical_data_list = _update_arrays(
                all_risk_scores, all_censorships, all_event_times, all_clinical_data,
                event_time, censor, risk, clinical_data_list)
            all_logits.append(h.detach().cpu().numpy())
            total_loss += loss_value
            all_slide_ids.append(slide_ids.values[count])
            count += 1

    total_loss /= len(loader.dataset)
    all_risk_scores       = np.concatenate(all_risk_scores, axis=0)
    all_risk_by_bin_scores= np.concatenate(all_risk_by_bin_scores, axis=0)
    all_censorships       = np.concatenate(all_censorships, axis=0)
    all_event_times       = np.concatenate(all_event_times, axis=0)
    all_logits            = np.concatenate(all_logits, axis=0)
    
    patient_results = {}
    for i in range(len(all_slide_ids)):
        slide_id = slide_ids.values[i]
        case_id  = slide_id[:12]
        patient_results[case_id] = {}
        patient_results[case_id]["time"]        = all_event_times[i]
        patient_results[case_id]["risk"]        = all_risk_scores[i]
        patient_results[case_id]["censorship"]  = all_censorships[i]
        patient_results[case_id]["clinical"]    = all_clinical_data[i]
        patient_results[case_id]["logits"]      = all_logits[i]
    
    c_index, c_index2, BS, IBS, iauc = _calculate_metrics(
        loader, dataset_factory, survival_train,
        all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores)

    return patient_results, c_index, c_index2, BS, IBS, iauc, total_loss

def _get_lr_scheduler(args, optimizer, dataloader):
    scheduler_name = args.lr_scheduler
    warmup_epochs  = args.warmup_epochs
    epochs = args.max_epochs if hasattr(args, 'max_epochs') else args.epochs

    warmup_steps = warmup_epochs * len(dataloader) if warmup_epochs > 0 else 0

    if scheduler_name == 'constant':
        lr_scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps)
    elif scheduler_name == 'cosine':
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=len(dataloader) * epochs)
    elif scheduler_name == 'linear':
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=len(dataloader) * epochs)
    return lr_scheduler

def _step(cur, args, loss_fn, model, optimizer, scheduler, train_loader, val_loader, test_loader):
    all_survival = _extract_survival_metadata(train_loader, val_loader, test_loader)
    best_loss = float('inf')

    for epoch in range(args.max_epochs):
        _train_loop_survival(epoch, model, args.modality, train_loader, optimizer, scheduler, loss_fn, args)
        _, val_cindex, _, _, _, _, total_loss = _summary(args.dataset_factory, model, args.modality, val_loader, loss_fn, all_survival, args.input_modality)

        print('Val loss:', total_loss, ', val_c_index:', val_cindex)
        if total_loss < best_loss:
            print("loss: {} -> {}".format(best_loss, total_loss))
            best_loss = total_loss
            print("Saving model...")
            torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    print("Loading model...")
    model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)), weights_only=True))

    print("Testing...")
    results_dict, test_cindex, test_cindex_ipcw, test_BS, test_IBS, test_iauc, total_loss = _summary(
        args.dataset_factory, model, args.modality, test_loader, loss_fn, all_survival, args.input_modality)

    print('Final Test c-index: {:.4f}'.format(test_cindex))
    return results_dict, (test_cindex, test_cindex_ipcw, test_BS, test_IBS, test_iauc, total_loss)

def _train_val_test(datasets, cur, args):
    train_split, val_split, test_split = _get_splits(datasets, cur, args)
    loss_fn    = _init_loss_function(args)
    model      = _init_model(args)
    optimizer  = _init_optim(args, model)
    train_loader, val_loader, test_loader = _init_loaders(args, train_split, val_split, test_split)
    lr_scheduler = _get_lr_scheduler(args, optimizer, train_loader)
    results_dict, (test_cindex, test_cindex2, test_BS, test_IBS, test_iauc, total_loss) = _step(
        cur, args, loss_fn, model, optimizer, lr_scheduler, train_loader, val_loader, test_loader)
    return results_dict, (test_cindex, test_cindex2, test_BS, test_IBS, test_iauc, total_loss)

def _test(datasets, cur, args):
    train_split, val_split, test_split = _get_splits(datasets, cur, args)
    loss_fn = _init_loss_function(args)
    model   = _init_model(args)
    model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)), weights_only=True))
    train_loader, val_loader, test_loader = _init_loaders(args, train_split, val_split, test_split)
    all_survival = _extract_survival_metadata(train_loader, val_loader, test_loader)
    results_dict, test_cindex, test_cindex_ipcw, test_BS, test_IBS, test_iauc, total_loss = _summary(
        args.dataset_factory, model, args.modality, test_loader, loss_fn,
        survival_train=all_survival, input_modality=args.test_modality, args=args)
    print('Final Test c-index: {:.4f}'.format(test_cindex))
    return results_dict, (test_cindex, test_cindex_ipcw, test_BS, test_IBS, test_iauc, total_loss)