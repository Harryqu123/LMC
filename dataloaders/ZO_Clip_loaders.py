from torch.utils.data import DataLoader, Dataset
import numpy as np
import os, json
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
from PIL import Image
from torchvision.datasets import ImageFolder
import glob
from torch.utils.data import Subset
try:
    from dataloaders.split import splits_2020
except:
    from split import splits_2020


def tiny_single_isolated_class_dino_loader(root="data/dalle_generated/union_set",
    tiny_dino_labels=['African_elephant', 'American_lobster', 'Christmas_stocking', 'European_fire_salamander', 'Labrador_retriever', 'Persian_cat', 'abacus', 'academic_gown', 'accordion', 'acorn', 'airplane', 'altar', 'ambulance', 'anemone', 'apple', 'arch_bridge', 'barn', 'barnacles', 'barrel', 'basketball', 'beaker', 'bee', 'beetle', 'bell_pepper', 'bighorn', 'binoculars', 'birdhouse', 'bison', 'black_stork', 'blender', 'book', 'bowling_ball', 'brain_coral', 'brass', 'brown_bear', 'bucket', 'bullet_train', 'burger', 'butcher_shop', 'cable-stayed_bridge', 'cable_car', 'cactus', 'canyon', 'cappuccino', 'cardigan', 'carriage', 'carrot', 'cart', 'castanets', 'cat', 'cattle_prod', 'chameleon', 'cheeseburger', 'chest', 'chicken_coop', 'choir_robe', 'christmas_tree', 'church', 'cliff_dwelling', 'clock', 'coconut', 'coffee_mug', 'coffee_table', 'computer_keyboard', 'confectionery', 'console_table', 'copper', 'coral_reef', 'corkscrew', 'cowboy_hat', 'crane', 'crate', 'cruise_ship', 'cucumber', 'cuttlefish', 'daisy', 'dandelion', 'digital_clock', 'dining_chair', 'dining_table', 'dobro', 'dog', 'dolphin', 'dragonfly', 'drumstick', 'dugong', 'dumbbell', 'dumbwaiter', 'eagle', 'elephant', 'envelope', 'espresso', 'fire_hydrant', 'fire_truck', 'flagstaff', 'flamingo', 'fleece', 'flip-flop', 'fly', 'flying_saucer', 'fountain', 'fox', 'frisbee', 'funicular', 'giraffe', 'gloves', 'golden_retriever', 'goldfish', 'golf_ball', 'gondola', 'goose', 'grasshopper', 'grizzly_bear', 'guitar', 'handcuffs', 'harp', 'hat', 'hay_bale', 'head_cabbage', 'hedge_trimmer', 'helicopter', 'hill', 'holly', 'horse', 'hourglass', 'jellyfish', 'jigsaw_puzzle', 'judge_robe', 'kangaroo', 'kayak', 'king_penguin', 'ladybug', 'lakeside', 'lamp', 'latte', 'lawn_mower', 'lemon', 'lesser_panda', 'letter_opener', 'lion', 'lipstick', 'macchiato', 'mailbox', 'manatee', 'mandolin', 'mantis', 'measuring_spoon', 'meat_loaf', 'megaphone', 'microscope', 'military_uniform', 'milk_can', 'monarch', 'mongoose', 'monorail', 'mosque', 'moss', 'motorbike', 'motorcycle', 'mountain', 'mountain_bike', 'neck_brace', 'nurse_uniform', 'ocarina', 'ocean', 'octopus', 'oil_lamp', 'orange', 'oscilloscope', 'otter', 'paintbrush', 'pancake', 'papaya', 'peach', 'peacock', 'periscope', 'piano_bench', 'pillow', 'pinecone', 'pipe_wrench', 'pizza', 'plastic_bag', 'plate', 'police_van', 'pomegranate', 'poncho', 'popsicle_stick', 'power_drill', 'punching_bag', 'quesadilla', 'red_panda', 'refrigerator', 'river', 'rooster', 'rugby_ball', 'saddlebag', 'sailboat', 'sandal', 'sandwich', 'saxophone', 'scarf', 'school_bus', 'scorpion', 'scuba_diver', 'sea_anemone', 'sea_cucumber', 'sea_slug', 'sea_urchin', 'seahorse', 'seashore', 'seaweed', 'seesaw', 'sewing_machine', 'sewing_needle', 'shawl', 'shower_cap', 'side_table', 'silverware', 'sleeping_bag', 'snail', 'snowmobile', 'soccer_ball', 'spider_web', 'spiny_lobster', 'sports_car', 'standard_poodle', 'starfish', 'stingray', 'stopwatch', 'street_sign', 'subway_car', 'subway_train', 'sushi_roll', 'suspension_bridge', 'sweater', 'swimming_trunks', 'swordfish', 'table_lamp', 'tape_player', 'teddy', 'telescope', 'temple', 'tennis_ball', 'thatch', 'thatched_roof', 'thimble', 'tiger', 'toaster', 'tomato', 'tortilla', 'toy_poodle', 'traffic_light', 'tram', 'trash_can', 'tree_frog', 'trilobite', 'trolleybus', 'trophy', 'trowel', 'truss_bridge', 'turnstile', 'umbrella', 'vacuum_cleaner', 'vase', 'vestment', 'viaduct', 'violin_case', 'volleyball', 'washbasin', 'water_tower', 'waterfall', 'watering_can', 'watermelon', 'whale', 'whale_shark', 'wheelbarrow', 'window_screen', 'wine_barrel', 'wok', 'wolf', 'wooden_crate', 'wooden_spoon']
                                           ):
    loaders_dict = {}
    transform = Compose([
        Resize(224, interpolation=3),
        CenterCrop(224),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    for class_label in tiny_dino_labels:
       dataset = ImageFolder(root=root, transform=transform)
       idx = [i for i in range(len(dataset)) if dataset.imgs[i][1] == dataset.class_to_idx[class_label]]
       # build the appropriate subset
       subset = Subset(dataset, idx)
       loader = DataLoader(dataset=subset, batch_size=1, num_workers=4)
       loaders_dict[class_label] = loader
    return loaders_dict, tiny_dino_labels

def cifar10_single_isolated_class_dino_loader(root="data/dalle_generated/union_set", labels=[]):
    loaders_dict = {}
    transform = Compose([
        Resize(224, interpolation=3),
        CenterCrop(224),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    for class_label in labels:
       dataset = ImageFolder(root = root, transform = transform)
       idx = [i for i in range(len(dataset)) if dataset.imgs[i][1] == dataset.class_to_idx[class_label]]
       # build the appropriate subset
       subset = Subset(dataset, idx)
       loader = DataLoader(dataset=subset, batch_size=1, num_workers=4)
       loaders_dict[class_label] = loader
    return loaders_dict, labels

def cifarplus_single_isolated_class_dino_loader(root="data/dalle_generated/union_set", labels=[]):
    loaders_dict = {}
    transform = Compose([
        Resize(224, interpolation=3),
        CenterCrop(224),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    for class_label in labels:
       dataset = ImageFolder(root = root, transform = transform)
       idx = [i for i in range(len(dataset)) if dataset.imgs[i][1] == dataset.class_to_idx[class_label]]
       # build the appropriate subset
       subset = Subset(dataset, idx)
       loader = DataLoader(dataset=subset, batch_size=1, num_workers=4)
       loaders_dict[class_label] = loader
    return loaders_dict


def tinyimage_get_all_labels():
    dataset = ImageFolder(root='./data/tiny-imagenet-200/val')
    a = dataset.class_to_idx
    reverse_a = {v:k for k,v in a.items()}
    all = list(dataset.class_to_idx.keys())
    f = open('./dataloaders/imagenet_id_to_label.txt', 'r')
    imagenet_id_idx_semantic = f.readlines()
    all_labels = []
    for id in all:
        for line in imagenet_id_idx_semantic:
            if id == line[:-1].split(' ')[0]:
                semantic_label = line[:-1].split(' ')[2]
                all_labels.append(semantic_label)
                break

    return all_labels


def tinyimage_semantic_spit_generator():
    tinyimage_splits = splits_2020["tiny_imagenet"]
    dataset = ImageFolder(root='./data/tiny-imagenet-200/val')
    a = dataset.class_to_idx
    reverse_a = {v:k for k,v in a.items()}
    semantic_splits = [[],[],[],[],[]]
    for i, split in enumerate(tinyimage_splits):
       wnid_split = []
       for idx in split:
           wnid_split.append(reverse_a[idx])
       all = list(dataset.class_to_idx.keys())
       seen = wnid_split
       unseen = list(set(all)-set(seen))
       seen.extend(unseen)
       f = open('./dataloaders/imagenet_id_to_label.txt', 'r')
       imagenet_id_idx_semantic = f.readlines()

       for id in seen:
           for line in imagenet_id_idx_semantic:
               if id == line[:-1].split(' ')[0]:
                   semantic_label = line[:-1].split(' ')[2]
                   semantic_splits[i].append(semantic_label)
                   break
    semantic_splits_detail = {}
    imagenet_id_idx_semantic_detail = json.load(open('dataloaders/tinyimagenet_labels.json'))
    for line in imagenet_id_idx_semantic:
        id = line[:-1].split(' ')[0]
        if id in seen:
            semantic_label = line[:-1].split(' ')[2]
            semantic_splits_detail[semantic_label] = imagenet_id_idx_semantic_detail[id]

    return semantic_splits, semantic_splits_detail


class tinyimage_isolated_class(Dataset):
    def __init__(self, label, mappings, train=False):
        assert label, 'a semantic label should be specified'
        super(tinyimage_isolated_class, self).__init__()
        if train:
            path = './data/tiny-imagenet-200/train/'
        else:
            path = './data/tiny-imagenet-200/val/'
        #path = '/Users/Sepid/data/tiny-imagenet-200/val/'
        self.image_paths = glob.glob(os.path.join(path, mappings[label], '*.JPEG'))
        self.transform = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        x = Image.open(self.image_paths[index]).convert('RGB')
        if self.transform:
            x = self.transform(x)
        return x


def tinyimage_single_isolated_class_loader(train=False):
    semantic_splits, semantic_splits_detail = tinyimage_semantic_spit_generator()
    f = open('./dataloaders/tinyimagenet_labels_to_ids.txt', 'r')
    #f = open('../tinyimagenet_ids_to_label.txt', 'r')
    tinyimg_label2folder = f.readlines()
    mappings_dict = {}
    for line in tinyimg_label2folder:
        label, class_id = line[:-1].split(' ')[0], line[:-1].split(' ')[1]
        mappings_dict[label] = class_id

    loaders_dict = {}
    if not train:
        for semantic_label in mappings_dict.keys():
            dataset = tinyimage_isolated_class(semantic_label, mappings_dict)
            loader = DataLoader(dataset=dataset, batch_size=1, num_workers=4)
            loaders_dict[semantic_label] = loader
        return semantic_splits, semantic_splits_detail, loaders_dict
    else:
        all_seen_labels = []
        for split in semantic_splits:
            all_seen_labels += split[:20]
        all_seen_labels = list(set(all_seen_labels))
        for semantic_label in all_seen_labels:
            dataset = tinyimage_isolated_class(semantic_label, mappings_dict)
            loader = DataLoader(dataset=dataset, batch_size=1, num_workers=4)
            loaders_dict[semantic_label] = loader
        return all_seen_labels, loaders_dict

class cifar10_isolated_class(Dataset):
    def __init__(self, class_label=None):
        assert class_label, 'a semantic label should be specified'
        super(cifar10_isolated_class, self).__init__()
        self.transform = Compose([
            ToPILImage(),
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615))
        ])
        cifar10 = CIFAR10(root='./data', train=False, download=True)

        class_mask = np.array(cifar10.targets) == cifar10.class_to_idx[class_label]
        self.data = cifar10.data[class_mask]
        self.targets = np.array(cifar10.targets)[class_mask]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.transform(self.data[index])


def cifar10_single_isolated_class_loader(batch_size=1):
    loaders_dict = {}
    cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    for label in cifar10_labels:
       dataset = cifar10_isolated_class(label)
       loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4)
       loaders_dict[label] = loader
    return loaders_dict

class cifarplus():
    def __init__(self, class_list):
        self.cifar100_classes = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 
                                 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 
                                 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 
                                 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
                                 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 
                                 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 
                                 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 
                                 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck',
                                 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 
                                 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 
                                 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 
                                 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 
                                 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 
                                 'woman', 'worm']
        self.class_list = class_list
        self.transform = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224), 
            ToTensor(),
            Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615))
        ])

        if len(self.class_list) == 4:
           cifar10 = CIFAR10(root='./data', train=False, download=True, transform=self.transform)
           inds = [i for i in range(len(cifar10.targets)) if cifar10.targets[i] in self.class_list]
           self.data = cifar10.data[inds]
           self.targets = np.array(cifar10.targets)[inds].tolist()
        else:
            cifar100 = CIFAR100(root='./data', train=False, download=True, transform=self.transform)
            inds = [i for i in range(len(cifar100.targets)) if cifar100.targets[i] in self.class_list]
            self.data = cifar100.data[inds]
            self.targets = np.array(cifar100.targets)[inds].tolist()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img = self.transform(Image.fromarray(self.data[index]).convert('RGB'))
        return img, self.targets[index]
    
def cifarplus_loader():

    cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    seen_labels = {}

    in_dict = {}
    for idx, list_ in enumerate(splits_2020['cifar100']):
        list_ = sorted(list_)
        in_dict['seen-{}'.format(str(idx+1))] = list_
        seen_labels['seen-{}'.format(str(idx+1))] = [cifar10_labels[i] for i in list_]

    out_dict = {}
    plus10_split = splits_2020['cifar100-10']
    plus50_split = splits_2020['cifar100-50']

    for idx, split in enumerate(plus10_split):
        out_dict['cifar100-10-{}'.format(str(idx+1))] = split
    for idx, split in enumerate(plus50_split):
        out_dict['cifar100-50-{}'.format(str(idx+1))] = split

    in_loaders = {}
    for key, list in in_dict.items():
        in_dataset = cifarplus(list)
        in_loader = DataLoader(dataset=in_dataset, batch_size=1, num_workers=4, shuffle=False)
        in_loaders[key] = in_loader


    out_loaders = {}
    for key in out_dict.keys():
        out_dataset = cifarplus(out_dict[key])
        out_loaders[key] = DataLoader(dataset=out_dataset, batch_size=1, num_workers=4, shuffle=False)

    return in_loaders, out_loaders, seen_labels