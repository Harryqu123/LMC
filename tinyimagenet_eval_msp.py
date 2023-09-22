import argparse
import torch
import os
from dataloaders.ZO_Clip_loaders import tinyimage_single_isolated_class_loader, tiny_single_isolated_class_dino_loader, tinyimage_semantic_spit_generator
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
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser("tiny imagenet eval")
parser.add_argument("--gpu_devices", default=1, help="gpu device")
parser.add_argument("--k_images", default=10, type=int)
parser.add_argument("--save_dir", default="output/tiny_imagenet")
parser.add_argument("--image_path", default="output/self_debug/keep")



def image_decoder(clip_model, dino_model, stored_features, k_images, device, image_loaders, split, chatgpt_manual_similar_label, detailed_labels=None):
    seen_labels = split[:20]
    if detailed_labels is not None:
        seen_descriptions = [f"This is a photo of a {detailed_labels[label]}" for label in seen_labels]
    else:
        seen_descriptions = [f"This is a photo of a {label}" for label in seen_labels]

    n_seen = sum([len(image_loaders[label]) for label in seen_labels])
    n_unseen = sum([len(image_loaders[label]) for label in split[20:]])
    targets = torch.tensor(n_seen*[0] + n_unseen*[1])
    

    clip_ood_probs_sum = []
    dino_ood_probs_sum = []
    clip_closeset_probs_sum = []
    dino_closeset_probs_sum = []
    closeset_labels_list = []
    for i, semantic_label in tqdm(enumerate(split)):
        if semantic_label in seen_labels:
            close_set = True
        else:
            close_set = False
        loader = image_loaders[semantic_label]

        # Prepare for dino
        total_labels = seen_labels + chatgpt_manual_similar_label
        total_features = []
        for i in total_labels:
            if i in stored_features: 
                feats = stored_features[i]
            else: 
                feats = None
            if feats is not None:
                if feats.shape[0] < k_images:
                    k = feats.shape[0]
                    k_short = k_images - k
                    n = k_short // k
                    p = k_short % k
                    stack_feat = [feats for n_ in range(n + 1)]
                    stack_feat.append(feats[:p, ...])
                    feats = torch.cat(stack_feat, dim=0)
                    assert feats.shape[0] == k_images
                total_features.append(feats)
            else:
                if i in seen_labels: 
                    raise NotImplementedError("no image for class {}".format(i))
        total_features = torch.cat(total_features, dim=0)
        total_features = total_features.t() # (d, k_images * k_class)

        for idx, image in enumerate(loader):
            
            # CLIP Alignment
            all_desc = seen_descriptions + [f"This is a photo of a {label}" for label in chatgpt_manual_similar_label]
            all_desc_ids = tokenize_for_clip(all_desc, cliptokenizer)

            with torch.no_grad():
                image_feature = clip_model.encode_image(image.cuda()).float()
                image_feature /= image_feature.norm(dim=-1, keepdim=True)
                text_features = clip_model.encode_text(all_desc_ids.cuda()).float()
                text_features /= text_features.norm(dim=-1, keepdim=True)
            zeroshot_probs = (100.0 * image_feature @ text_features.T).softmax(dim=-1).squeeze()

            clip_ood_prob_sum = zeroshot_probs[:20].detach().cpu().numpy()
            clip_ood_probs_sum.append(clip_ood_prob_sum)

            # DINO Alignment
            with torch.no_grad():
                image = image.cuda()
                feats = dino_model(image)
                feats = torch.nn.functional.normalize(feats, dim=1, p=2)

            # softmax then take sum (avg) for each class
            zeroshot_probs_dino = (100.0 * feats @ total_features)
            zeroshot_probs_dino_cls = zeroshot_probs_dino.split(k_images, dim=-1)
            zeroshot_probs_dino_cls = torch.tensor([torch.mean(zeroshot_probs_dino_cls[i]) for i in range(len(total_labels))]).softmax(dim=-1).squeeze()
            ood_prob_sum_dino = zeroshot_probs_dino_cls[:len(seen_labels)].detach().cpu().numpy()
            dino_ood_probs_sum.append(ood_prob_sum_dino)

            if close_set:
                # CLIP
                with torch.no_grad():
                    seen_desc_ids = tokenize_for_clip(seen_descriptions, cliptokenizer)
                    seen_text_feature = clip_model.encode_text(seen_desc_ids.cuda()).float()
                    seen_text_feature /= seen_text_feature.norm(dim=-1, keepdim=True)
                clip_closeset_probs = (100.0 * image_feature @ seen_text_feature.T).softmax(dim=-1).squeeze()
                clip_closeset_probs_sum.append(clip_closeset_probs.detach().cpu().numpy())
                closeset_labels_list.append(seen_labels.index(semantic_label))

                # DINO
                closeset_probs_dino = (100.0 * feats @ total_features[:, :len(seen_labels*k_images)])
                closeset_probs_dino_per_cls_ = closeset_probs_dino.split(k_images, dim=1)
                closeset_probs_dino_per_cls = torch.tensor([torch.mean(closeset_probs_dino_per_cls_[i]) for i in range(len(seen_labels))])
                closeset_probs_dino = closeset_probs_dino_per_cls.softmax(dim=-1).squeeze()
                dino_closeset_probs_sum.append(closeset_probs_dino.detach().cpu().numpy())


    prob = 0.6
    ood_probs_sum_ = [a * prob + b * (1 - prob) for (a, b) in zip(clip_ood_probs_sum, dino_ood_probs_sum)]
    ood_probs_sum = [1 - max(ood_probs_sum_[ii]) for ii in range(len(ood_probs_sum_))]
    auc_sum = roc_auc_score(np.array(targets), np.squeeze(ood_probs_sum))

    closeset_probs_sum = [a * prob + b * (1 - prob) for (a, b) in zip(clip_closeset_probs_sum, dino_closeset_probs_sum)]

    closeset_preds_list = []
    for closeset_prob in closeset_probs_sum:
        closeset_pred = np.argmax(closeset_prob, axis=-1)
        closeset_pred_label = seen_labels[closeset_pred]
        closeset_preds_list.append(seen_labels.index(closeset_pred_label))
    closeset_preds_list = np.array(closeset_preds_list)
    closeset_labels_list = np.array(closeset_labels_list)
    oscr = compute_oscr(np.squeeze(ood_probs_sum)[:n_seen], np.squeeze(ood_probs_sum)[n_seen:], closeset_preds_list, closeset_labels_list)
    print('AUROC = {}, OSCR = {}'.format(auc_sum, oscr))


    return auc_sum, oscr

if __name__ == '__main__':
    args = parser.parse_args()
    k_images = args.k_images
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    time_str = datetime.strftime(datetime.now(), '%Y-%m-%d-%H:%M:%S')
    sys.stdout = Logger(os.path.join(args.save_dir, 'eval_{}.log'.format(time_str)))
    print('settings:')
    print(args)

    # prepare dino model
    dino_model = torch.hub.load('facebookresearch/dinov2', "dinov2_vitb14")
    dino_model.cuda()

    state_dict = torch.load("pretrained_model/dinov2_vitb14_pretrain.pth", map_location='cpu')
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = dino_model.load_state_dict(state_dict, strict=False)
    print('Pretrained weights found at pretrained_model/dinov2_vitb14_pretrain.pth and loaded with msg: {}'.format(msg))
    dino_model.eval()

    # prepare clip model
    # initialize tokenizers for clip and bert, these two use different tokenizers
    clip_model = torch.jit.load("pretrained_model/ViT-B-32.pt").to(device).eval()
    cliptokenizer = clip_tokenizer()

    chatgpt_dict = json.load(open('chat_json/tinyimagenet.json'))

    chatgpt_labels = []
    virtual_labels = []
    for i in range(5):
        virtual_labels.append(chatgpt_dict[str(i)])
        chatgpt_labels += chatgpt_dict[str(i)]
    chatgpt_labels = list(set(chatgpt_labels))

    all_seen_labels = []
    semantic_splits, _ = tinyimage_semantic_spit_generator()
    for split in semantic_splits:
        all_seen_labels += split[:20]
    all_seen_labels = list(set(all_seen_labels))

    stored_features_list = []
    for i in range(5):
        labels = virtual_labels[i]
        labels += semantic_splits[i][:20]
        image_root = os.path.join(args.image_path, str(i))
        classes = os.listdir(image_root)
        for l in labels:
            if l not in classes:
                labels.remove(l)
        tiny_dino_loaders, tiny_dino_labels = tiny_single_isolated_class_dino_loader(tiny_dino_labels=labels, root=image_root)
        stored_features = {}
        for idx_lable, semantic_label in enumerate(tiny_dino_labels):
            print("Extracting features {} {}/{}".format(semantic_label, idx_lable, len(tiny_dino_labels)))
            if semantic_label not in tiny_dino_loaders:
                continue
            stored_features[semantic_label] = extract_features(dino_model, tiny_dino_loaders[semantic_label], k_images)
        print("Finish storing features")
        stored_features_list.append(stored_features)


    splits, detailed_labels, tinyimg_loaders = tinyimage_single_isolated_class_loader()
    print('seen splits:')
    for split in splits:
        print(split[:20])
    
    auc_scores = []
    oscr_scores = []

    for index, split in enumerate(splits):
        chatgpt_labels = virtual_labels[index]
        stored_features = stored_features_list[index]
        auc_list_sum_per_split, oscr_list_sum_per_split = image_decoder(clip_model=clip_model,
                                                                                    dino_model=dino_model, 
                                                                                    stored_features=stored_features,
                                                                                    k_images=k_images,
                                                                                    device=device, 
                                                                                    image_loaders=tinyimg_loaders,
                                                                                    split=split, 
                                                                                    chatgpt_manual_similar_label=chatgpt_labels,
                                                                                    detailed_labels=None)

        auc_scores.append(auc_list_sum_per_split)
        oscr_scores.append(oscr_list_sum_per_split)

    prob = 0.6

    print('Average over 5 splits:')
    print(' AUROC: {} +/- {}, {}'.format(np.mean(auc_scores), np.std(auc_scores), auc_scores))
    print(' OSCR: {} +/- {}, {}'.format(np.mean(oscr_scores), np.std(oscr_scores), oscr_scores))
