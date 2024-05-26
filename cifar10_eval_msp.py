import argparse
import torch
import os
from tqdm import tqdm
import numpy as np
from dataloaders.ZO_Clip_loaders import cifar10_single_isolated_class_dino_loader, cifar10_single_isolated_class_loader
from clip.simple_tokenizer import SimpleTokenizer as clip_tokenizer
from sklearn.metrics import roc_auc_score
import json
from dataloaders.split import splits_2020
from datetime import datetime
import sys
from utils_.utils_ import Logger, compute_oscr
from utils_.clip_utils import tokenize_for_clip
from utils_.dino_utils import extract_features
import warnings
warnings.filterwarnings("ignore")

cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


parser = argparse.ArgumentParser()
parser.add_argument("--gpu_device", default='1', type=str)
parser.add_argument("--k_images", default=10)
parser.add_argument("--save_dir", default="output/cifar_10")
parser.add_argument("--image_path", default="path/to/your/cifar/images")


def image_decoder(clip_model, dino_model, stored_features, k_images, device, image_loaders=None, image_path=""):
    auc_list_sum = []
    oscr_list_sum = []

    splits_idx = splits_2020['cifar10']
    splits = []
    for split_idx in splits_idx:
        unseens = list(set(range(10)) - set(split_idx))
        s = [cifar10_labels[idx] for idx in split_idx + unseens]
        splits.append(s)
    print("evaluating splits: ")
    print(splits)


    gpt_json = "path/to/your/chatgpt_generated_list.json"
    chatgpt_virtual_labels = json.load(open(gpt_json))

    for index, split in enumerate(splits):
        image_root = os.path.join(image_path, str(i))
        
        seen_labels = split[:6]
        unseen_labels = split[6:]
        n_seen = sum([len(image_loaders[label]) for label in seen_labels])
        n_unseen = sum([len(image_loaders[label]) for label in unseen_labels])
        targets = torch.tensor(n_seen*[0] + n_unseen*[1])
        

        # pre-store dino features
        virtual_classes = chatgpt_virtual_labels[str(index)]
        total_labels = seen_labels + virtual_classes
        cifar10_dino_loaders, cifar10_dino_labels = cifar10_single_isolated_class_dino_loader(root=image_root, labels=total_labels)
        stored_features = {}
        for idx_lable, semantic_label in enumerate(total_labels):
            print("Extracting features {} {}/{}".format(semantic_label, idx_lable, len(total_labels)))
            stored_features[semantic_label] = extract_features(dino_model, cifar10_dino_loaders[semantic_label], k_images=k_images)
        
        print("Finish storing features")

        
        # Prepare for dino
        total_features = []
        for i in total_labels:
            feats = stored_features[i]
            if feats.shape[0] < k_images:
                n = k_images - feats.shape[0]
                zero_feat = torch.zeros((n, feats.shape[1]), device=feats.device)
                feats = torch.cat([feats, zero_feat], dim=0)
            total_features.append(feats)
        total_features = torch.cat(total_features, dim=0)
        total_features = total_features.t()

        # Prepare for clip
        seen_descriptions = [f"This is a photo of a {label}" for label in seen_labels]

        ood_probs_sum_dino = []
        ood_probs_sum_clip = []
        closeset_preds_list = []
        closeset_preds_list_clip = []
        closeset_preds_list_dino = []
        closeset_labels_list = []
        for i, semantic_label in enumerate(split):
            print(semantic_label)
            if semantic_label in seen_labels:
                close_set = True
            else:
                close_set = False
            loader = image_loaders[semantic_label]
            for idx, samples in enumerate(loader):
                # DINO Alignment
                with torch.no_grad():
                    samples = samples.cuda()
                    feats = dino_model(samples)
                    feats = torch.nn.functional.normalize(feats, dim=1, p=2)
                zeroshot_probs_dino = (100.0 * feats @ total_features)# .softmax(dim=-1).squeeze()
                zeroshot_probs_dino_cls = zeroshot_probs_dino.split(k_images, dim=1)
                zeroshot_probs_dino_cls = torch.tensor([torch.mean(zeroshot_probs_dino_cls[i]) for i in range(len(total_labels))]).softmax(dim=-1).squeeze()
                ood_prob_sum_dino = zeroshot_probs_dino_cls[:len(seen_labels)].detach().cpu().numpy()
                ood_probs_sum_dino.append(ood_prob_sum_dino)
                

                # CLIP Alignment
                all_desc = seen_descriptions + [f"This is a photo of a {label}" for label in virtual_classes]
                all_desc_ids = tokenize_for_clip(all_desc, cliptokenizer)

                with torch.no_grad():
                    image_feature = clip_model.encode_image(samples.cuda()).float()
                    image_feature /= image_feature.norm(dim=-1, keepdim=True)
                    text_features = clip_model.encode_text(all_desc_ids.cuda()).float()
                    text_features /= text_features.norm(dim=-1, keepdim=True)


                zeroshot_probs_clip = (100.0 * image_feature @ text_features.T).softmax(dim=-1).squeeze() 
                ood_prob_sum_clip = zeroshot_probs_clip[:len(seen_labels)].detach().cpu().numpy()
                ood_probs_sum_clip.append(ood_prob_sum_clip)
                
                if close_set:
                    # CLIP
                    with torch.no_grad():
                        seen_desc_ids = tokenize_for_clip(seen_descriptions, cliptokenizer)
                        seen_text_feature = clip_model.encode_text(seen_desc_ids.cuda()).float()
                        seen_text_feature /= seen_text_feature.norm(dim=-1, keepdim=True)
                    closeset_probs_clip = (100.0 * image_feature @ seen_text_feature.T).softmax(dim=-1).squeeze()
                    closeset_preds_list_clip.append(closeset_probs_clip.detach().cpu().numpy())
                    
                    # DINO
                    closeset_probs_dino = (100.0 * feats @ total_features[:, :len(seen_labels*k_images)])
                    closeset_probs_dino_per_cls_ = closeset_probs_dino.split(k_images, dim=1)
                    closeset_probs_dino_per_cls = torch.tensor([torch.mean(closeset_probs_dino_per_cls_[i]) for i in range(len(seen_labels))])
                    closeset_probs_dino = closeset_probs_dino_per_cls.softmax(dim=-1).squeeze()
                    closeset_preds_list_dino.append(closeset_probs_dino.detach().cpu().numpy())
                    closeset_labels_list.append(cifar10_labels.index(semantic_label))

        



        prob = 0.6
        ood_probs_sum_ = [a * prob + b * (1 - prob) for (a, b) in zip(ood_probs_sum_clip, ood_probs_sum_dino)]
        ood_probs_sum = [1 - max(ood_probs_sum_[ii]) for ii in range(len(ood_probs_sum_))]
        auc_sum = roc_auc_score(np.array(targets), np.squeeze(ood_probs_sum))

        assert len(closeset_preds_list_clip) == n_seen
        closeset_probs_sum = [p_clip * prob + p_dino * (1 - prob) for (p_clip, p_dino) in zip(closeset_preds_list_clip, closeset_preds_list_dino)]
        closeset_preds_list = []
        for closeset_prob in closeset_probs_sum:
            closeset_pred = np.argmax(closeset_prob, axis=-1)
            closeset_pred_label = seen_labels[closeset_pred]
            closeset_preds_list.append(cifar10_labels.index(closeset_pred_label))
        closeset_preds_list = np.array(closeset_preds_list)
        closeset_labels_list = np.array(closeset_labels_list)
        oscr = compute_oscr(np.squeeze(ood_probs_sum)[:n_seen], np.squeeze(ood_probs_sum)[n_seen:], closeset_preds_list, closeset_labels_list)
        print('split {}, AUROC = {}, oscr = {}'.format(index, auc_sum, oscr))
        auc_list_sum.append(auc_sum)
        oscr_list_sum.append(oscr)

    print('Average over 5 splits:')
    print(' AUROC: {} +/- {}, {}'.format(np.mean(auc_list_sum), np.std(auc_list_sum), auc_list_sum))
    print(' OSCR: {} +/- {}, {}'.format(np.mean(oscr_list_sum), np.std(oscr_list_sum), oscr_list_sum))
    

if __name__ == '__main__':

    args = parser.parse_args()
    k_images = args.k_images
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    save_dir = args.save_dir
    image_root = args.image_path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    time_str = datetime.strftime(datetime.now(), '%Y-%m-%d-%H:%M:%S')
    sys.stdout = Logger(os.path.join(args.save_dir, 'eval_{}.log'.format(time_str)))
    print('settings:')
    print(args)


    device = torch.device('cuda:{}'.format('0') if torch.cuda.is_available() else 'cpu')

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

    cifar10_loaders_eval = cifar10_single_isolated_class_loader()
    
    image_decoder(clip_model, 
                  dino_model, 
                  stored_features=None, 
                  k_images=k_images,
                  device=device, 
                  image_loaders=cifar10_loaders_eval,
                  image_path=image_root)


