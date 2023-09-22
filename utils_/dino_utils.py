import torch

def extract_features(model, data_loader, k_images):
    features = []
    for index, items in enumerate(data_loader):
    # for index, (samples, targets) in enumerate(data_loader):
        # print(type(items))
        if isinstance(items, tuple):
            (samples, targets) = items
        elif isinstance(items, list) and len(items) == 2:
            (samples, targets) = items
        elif isinstance(items, list) and len(items) == 3:
            (samples, targets, _) = items
        elif torch.is_tensor(items):
            samples = items
        if index == k_images: break
        with torch.no_grad():
            samples = samples.cuda()
            feats = model(samples)
            features.append(feats)
        
    features = torch.cat(features, dim=0)
    features = torch.nn.functional.normalize(features, dim=1, p=2)
    print("Storing feature: {}".format(features.shape))
    return features