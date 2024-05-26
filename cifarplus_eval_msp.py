import argparse
import torch
import os
from dataloaders.ZO_Clip_loaders import cifarplus_loader, cifarplus_single_isolated_class_dino_loader
from clip.simple_tokenizer import SimpleTokenizer as clip_tokenizer
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
import json
from datetime import datetime
import sys
from utils_.utils_ import Logger, compute_oscr
from utils_.clip_utils import tokenize_for_clip
from utils_.dino_utils import extract_features

import warnings
warnings.filterwarnings("ignore")


cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

parser = argparse.ArgumentParser("cifar plus eval")
parser.add_argument("--gpu_devices", default=1, help="gpu device")
parser.add_argument("--k_images", default=10, type=int, help="number of images from dalle")
parser.add_argument("--save_dir", default="output/cifar_plus")
parser.add_argument("--image_path", default="path/to/your/cifarplus/images")


def image_decoder(clip_model, dino_model, chatgpt_labels, k_images, in_loader, out_loaders, seen_labels, stored_features, description="This is a photo of a {label}"):
    # Prepare for CLIP
    seen_descriptions = [description.format(label=label) for label in seen_labels]

    # Prepare for DINO
    total_labels = seen_labels + chatgpt_labels
    total_features = []
    for semantic_label in total_labels:
        feats = stored_features[semantic_label]
        if feats.shape[0] < k_images:
            n = k_images - feats.shape[0]
            zero_feat = torch.zeros((n, feats.shape[1]), device=feats.device)
            feats = torch.cat([feats, zero_feat], dim=0)
        total_features.append(feats)
    total_features = torch.cat(total_features, dim=0)
    total_features = total_features.t()

    in_probs_sum_dino = []
    in_probs_sum_clip = []
    closeset_probs_list_clip = []
    closeset_probs_list_dino = []
    closeset_labels_list = []
    for idx, (image, label_idx) in enumerate(in_loader):

        # CLIP Alignment
        all_desc = seen_descriptions + [description.format(label=label) for label in chatgpt_labels]
        all_desc_ids = tokenize_for_clip(all_desc, cliptokenizer)

        with torch.no_grad():
            image_feature = clip_model.encode_image(image.cuda()).float()
            image_feature /= image_feature.norm(dim=-1, keepdim=True)
            text_features = clip_model.encode_text(all_desc_ids.cuda()).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
        zeroshot_probs_clip = (100.0 * image_feature @ text_features.T).softmax(dim=-1).squeeze()

        in_prob_sum = zeroshot_probs_clip[:len(seen_labels)].detach().cpu().numpy()
        in_probs_sum_clip.append(in_prob_sum)


        # DINO Alignment
        with torch.no_grad():
            image = image.cuda()
            feats = dino_model(image)
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        zeroshot_probs_dino = (100.0 * feats @ total_features)
        zeroshot_probs_dino_cls_ = zeroshot_probs_dino.split(k_images, dim=1)
        zeroshot_probs_dino_cls = torch.tensor([torch.mean(zeroshot_probs_dino_cls_[i]) for i in range(len(total_labels))])
        zeroshot_probs_dino_cls = zeroshot_probs_dino_cls.softmax(dim=-1).squeeze()
        in_prob_sum_dino = zeroshot_probs_dino_cls[:len(seen_labels)].detach().cpu().numpy()
        in_probs_sum_dino.append(in_prob_sum_dino)

        # closed-set classification
        # CLIP
        with torch.no_grad():
            seen_desc_ids = tokenize_for_clip(seen_descriptions, cliptokenizer)
            seen_text_features = clip_model.encode_text(seen_desc_ids.cuda()).float()
            seen_text_features /= seen_text_features.norm(dim=-1, keepdim=True)
            closeset_probs_clip = (100.0 * image_feature @ seen_text_features.T).softmax(dim=-1).squeeze()
            closeset_probs_list_clip.append(closeset_probs_clip.detach().cpu().numpy())

            # DINO
            closeset_probs_dino = (100.0 * feats @ total_features[:, :len(seen_labels*k_images)])
            closeset_probs_dino_per_cls_ = closeset_probs_dino.split(k_images, dim=1)
            closeset_probs_dino_per_cls = torch.tensor([torch.mean(closeset_probs_dino_per_cls_[i]) for i in range(len(seen_labels))])
            closeset_probs_dino = closeset_probs_dino_per_cls.softmax(dim=-1).squeeze()
            closeset_probs_list_dino.append(closeset_probs_dino.detach().cpu().numpy())

            closeset_labels_list.append(label_idx.cpu().numpy())


    closeset_labels_list = np.concatenate(closeset_labels_list, axis=0)

    ood_probs_sum_list_clip = [[] for i in range(len(out_loaders))]
    ood_probs_sum_list_dino = [[] for i in range(len(out_loaders))]
    
    for i, out_loader_name in enumerate(list(out_loaders.keys())):
        print(out_loader_name)
        out_loader = out_loaders[out_loader_name]
        for idx, (image, label_idx) in enumerate(out_loader):
            
            # CLIP Alignment
            all_desc = seen_descriptions + [f"This is a photo of a {label}" for label in chatgpt_labels]
            all_desc_ids = tokenize_for_clip(all_desc, cliptokenizer)
            
            with torch.no_grad():
                image_feature = clip_model.encode_image(image.cuda()).float()
                image_feature /= image_feature.norm(dim=-1, keepdim=True)
                text_features = clip_model.encode_text(all_desc_ids.cuda()).float()
                text_features /= text_features.norm(dim=-1, keepdim=True)

            zeroshot_probs = (100.0 * image_feature @ text_features.T).softmax(dim=-1).squeeze()

            ood_prob_sum = zeroshot_probs[:len(seen_labels)].detach().cpu().numpy()
            ood_probs_sum_list_clip[i].append(ood_prob_sum)

            # DINO Alignment
            with torch.no_grad():
                image = image.cuda()
                feats = dino_model(image)
                feats = torch.nn.functional.normalize(feats, dim=1, p=2)
            zeroshot_probs_dino = (100.0 * feats @ total_features)
            zeroshot_probs_dino_cls_ = zeroshot_probs_dino.split(k_images, dim=1)
            zeroshot_probs_dino_cls = torch.tensor([torch.mean(zeroshot_probs_dino_cls_[i]) for i in range(len(total_labels))])
            zeroshot_probs_dino_cls = zeroshot_probs_dino_cls.softmax(dim=-1).squeeze()
            ood_prob_sum_dino = zeroshot_probs_dino_cls[:len(seen_labels)].detach().cpu().numpy()
            ood_probs_sum_list_dino[i].append(ood_prob_sum_dino)

            
    auc_results = {}
    oscr_results = {}
    for i, out_loader_name in enumerate(out_loaders.keys()):
    
        prob = 0.6
        out_loader = out_loaders[out_loader_name]
        targets = torch.tensor(len(in_loader.dataset)*[0] + len(out_loader.dataset)*[1])

        in_probs_sum_ = [a * prob + b * (1 - prob) for (a, b) in zip(in_probs_sum_clip, in_probs_sum_dino)]
        in_probs_sum = [1 - max(in_probs_sum_[iii]) for iii in range(len(in_probs_sum_))]
        ood_probs_sum_ = [a * prob + b * (1 - prob) for (a, b) in zip(ood_probs_sum_list_clip[i], ood_probs_sum_list_dino[i])]
        ood_probs_sum = [1 - max(ood_probs_sum_[iii]) for iii in range(len(ood_probs_sum_))]
        probs_sum = in_probs_sum + ood_probs_sum
        auc_sum = roc_auc_score(np.array(targets), np.squeeze(probs_sum))

        auc_results[out_loader_name] = auc_sum

        closeset_probs_sum = [p_clip * prob + p_dino * (1 - prob) for (p_clip, p_dino) in zip(closeset_probs_list_clip, closeset_probs_list_dino)]
        closeset_preds_list = []
        for closeset_prob in closeset_probs_sum:
            closeset_pred = np.argmax(closeset_prob, axis=-1)
            closeset_pred_label = seen_labels[closeset_pred]
            closeset_preds_list.append(cifar10_labels.index(closeset_pred_label))
        closeset_preds_list = np.array(closeset_preds_list)

        oscr = compute_oscr(np.array(in_probs_sum), np.array(ood_probs_sum), closeset_preds_list, closeset_labels_list)
        oscr_results[out_loader_name] = oscr
        print('dataset: {}, AUROC = {}, OSCR = {}, '.format(out_loader_name, auc_sum, oscr))

    return auc_results, oscr_results

if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = args.save_dir
    image_path = args.image_path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    time_str = datetime.strftime(datetime.now(), '%Y-%m-%d-%H:%M:%S')
    sys.stdout = Logger(os.path.join(args.save_dir, 'eval_{}.log'.format(time_str)))
    print("settings:")
    print(args)
    k_images = args.k_images

    dino_model = torch.hub.load('facebookresearch/dinov2', "dinov2_vitb14")
    dino_model = dino_model.to(device)
    state_dict = torch.load("pretrained_model/dinov2_vitb14_pretrain.pth", map_location='cpu')
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = dino_model.load_state_dict(state_dict, strict=False)
    print('Pretrained weights found at pretrained_model/dinov2_vitb14_pretrain.pth and loaded with msg: {}'.format(msg))
    dino_model.eval()

    clip_model = torch.jit.load("pretrained_model/ViT-B-32.pt").to(device).eval()
    cliptokenizer = clip_tokenizer()

    in_loaders, out_loaders, seen_labels = cifarplus_loader()
    print(seen_labels)

    chatgpt_dict = json.load(open('path/to/your/chatgpt_generated_list.json'))

    description = "This is a photo of a {label}"
    print(description.format(label='[cls]'))
    acc_closeset_list = []
    oscr_plus_10_list = []
    oscr_plus_50_list = []
    auc_plus_10_list = []
    auc_plus_50_list = []

    oscr_plus_10 = []
    oscr_plus_50 = []
    auc_plus_10 = []
    auc_plus_50 = []
    for idx, in_item in enumerate(in_loaders.items()):

        stored_features = {}
        in_key, current_in_loader = in_item
        cifar_plus_10_key = in_key.replace('seen', 'cifar100-10')
        cifar_plus_50_key = in_key.replace('seen', 'cifar100-50')
        cifar_plus_10_loader = out_loaders[cifar_plus_10_key]
        cifar_plus_50_loader = out_loaders[cifar_plus_50_key]
        current_out_loaders = {'cifar100-10': cifar_plus_10_loader, 'cifar100-50': cifar_plus_50_loader}
        current_seen_labels = seen_labels[in_key]
        current_virtual_classes = chatgpt_dict[str(idx)]


        image_root = os.path.join(image_path, str(idx))
        loaders = cifarplus_single_isolated_class_dino_loader(root=image_root, labels=current_seen_labels)
        for idx_lable, semantic_label in enumerate(current_seen_labels):
            print("Extracting features {} {}/{}".format(semantic_label, idx_lable, len(current_seen_labels)))
            loader = loaders[semantic_label]
            feats = extract_features(dino_model, loader, k_images)
            stored_features[semantic_label] = feats

        virtual_loaders = cifarplus_single_isolated_class_dino_loader(root="cifarplus_img/{}".format(str(idx)), labels=current_virtual_classes)
        for idx_lable, semantic_label in enumerate(current_virtual_classes):
            print("Extracting features {} {}/{}".format(semantic_label, idx_lable, len(current_virtual_classes)))
            loader = virtual_loaders[semantic_label]
            feats = extract_features(dino_model, loader, k_images)
            stored_features[semantic_label] = feats
        
        print("Finish storing features")

        auc_results, oscr_results = image_decoder(clip_model, 
                                                dino_model,
                                                chatgpt_labels=current_virtual_classes, 
                                                k_images=k_images,
                                                in_loader=current_in_loader, 
                                                out_loaders=current_out_loaders, 
                                                seen_labels=current_seen_labels,
                                                stored_features=stored_features,
                                                description=description)

        auc_plus_10_list.append(auc_results['cifar100-10'])
        auc_plus_50_list.append(auc_results['cifar100-50'])
        oscr_plus_10_list.append(oscr_results['cifar100-10'])
        oscr_plus_50_list.append(oscr_results['cifar100-50'])

    print('Average over 5 splits:')
    print("----------------------cifar100-10----------------------")
    print(' AUROC: {} +/- {}, {}'.format(np.mean(auc_plus_10_list), np.std(auc_plus_10_list), auc_plus_10_list))
    print(' OSCR: {} +/- {}, {}'.format(np.mean(oscr_plus_10_list), np.std(oscr_plus_10_list), oscr_plus_10_list))

    print("----------------------cifar100-50----------------------")
    print(' AUROC: {} +/- {}, {}'.format(np.mean(auc_plus_50_list), np.std(auc_plus_50_list), auc_plus_50_list))
    print(' OSCR: {} +/- {}, {}'.format(np.mean(oscr_plus_50_list), np.std(oscr_plus_50_list), oscr_plus_50_list))

    