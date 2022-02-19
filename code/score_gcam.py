import cv2
import pydicom
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision import models

from pytorch_grad_cam import GradCAM, LayerCAM
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import deprocess_image, preprocess_image


# Create mask to grounth truth
def create_mask(row):
    mask = np.zeros((1024, 1024), dtype=int)
    y = int(row['y'])
    y_max = int(row['y']+row['height']+1)
    x =  int(row['x'])
    x_max = int(row['x']+row['width']+1)
    mask[y:y_max+1, x:x_max] = 1
    return mask, (y, x), (y_max, x_max)


def load_img(image_path):
    dcm_file = pydicom.read_file(image_path)
    img_arr = dcm_file.pixel_array
    img = Image.fromarray(img_arr).convert('RGB')
    return img


def preprocess(img):
    img_raw = img.copy()
    if img_raw.size[0] > img_raw.size[1]:
        img_raw.thumbnail((1000000, 256))
    else:
        img_raw.thumbnail((256, 1000000))

    Left = (img_raw.width - 224) / 2
    Right = Left + 224
    Top = (img_raw.height - 244) / 2
    Buttom = Top + 224
    img_raw = img_raw.crop((Left, Top, Right, Buttom))

    return img_raw


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    x = heatmap
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam), x


# Gcam Method
def gcam(config, model, target_layers):
    methods = \
        {
            "gradcam": GradCAM,
            "layercam": LayerCAM
        }

    # rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    # rgb_img = np.float32(rgb_img) / 255
    # input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = load_img(config['root_path'] + config['img'] + ".dcm")
    rgb_img = preprocess(img)
    rgb_img = np.float32(rgb_img) / 255

    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = None

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    cam_algorithm = methods[config['method']]
    with cam_algorithm(model=model,
                       target_layers=target_layers,
                       use_cuda=config['use_cuda']) as cam:
        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32

        grayscale_cam = cam(input_tensor=input_tensor,
                            target_category=target_category,
                            aug_smooth=config['aug_smooth'],
                            eigen_smooth=config['eigen_smooth'])

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        cam_image, heatmap = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=config['use_cuda'])
    gb = gb_model(input_tensor, target_category=target_category)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    return cam_image, gb, cam_gb, heatmap, img


# Load trained model
def get_model(model_name, fold=None):
    """
      Choose the target layer you want to compute the visualization for.
      Usually this will be the last convolutional layer in the model.
      Some common choices can be:
      Resnet18 and 50: model.layer4[-1]
      VGG, densenet161: model.features[-1]
      mnasnet1_0: model.layers[-1]
      You can print the model to help chose the layer
      You can pass a list with several target layers,
      in that case the CAMs will be computed per layer and then aggregated.
      You can also try selecting all layers of a certain type, with e.g:
      from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
      find_layer_types_recursive(model, [torch.nn.ReLU])
    """

    if model_name == "resnet_50_raw":
        model = models.resnet50(pretrained=True)
        target_layers = [model.layer4[-1]]

    elif model_name == "densenet161_raw":
        model = models.densenet161(pretrained=True)
        target_layers = [model.features[-1]]

    elif model_name == "resnet_special_v2":
        model = torch.load('./models/resnet_fold0/checkpoint_10.pth.tar', map_location=torch.device('cpu'))['model']
        target_layers = [model.layer4[-1]]

    elif model_name == "densenet_special_v2":
        model = torch.load('./models/densenet_fold0/checkpoint_10.pth.tar', map_location=torch.device('cpu'))['model']
        target_layers = [model.features[-1]]

    elif model_name == "resnet_model_fold0":
        model = torch.load(f'./models/models/ResNet/res_fold0/checkpoint_45.pth.tar', map_location=torch.device('cpu'))['model']
        target_layers = [model.layer4[-1]]
        model.fc = torch.nn.Sequential(*list(model.fc.children())[:-1])

    elif model_name == "resnet_model_fold1":
        model = torch.load(f'./models/models/ResNet/res_fold1/checkpoint_46.pth.tar', map_location=torch.device('cpu'))['model']
        target_layers = [model.layer4[-1]]
        model.fc = torch.nn.Sequential(*list(model.fc.children())[:-1])

    elif model_name == "resnet_model_fold2":
        model = torch.load(f'./models/models/ResNet/res_fold2/checkpoint_42.pth.tar', map_location=torch.device('cpu'))['model']
        target_layers = [model.layer4[-1]]
        model.fc = torch.nn.Sequential(*list(model.fc.children())[:-1])

    elif model_name == "densenet_model_fold0":
        model = torch.load(f'./models/models/DenseNet/dense_fold0/checkpoint_10.pth.tar', map_location=torch.device('cpu'))['model']
        target_layers = [model.features[-1]]
        model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-1])

    elif model_name == "densenet_model_fold1":
        model = torch.load(f'./models/models/DenseNet/dense_fold1/checkpoint_10.pth.tar', map_location=torch.device('cpu'))['model']
        target_layers = [model.features[-1]]
        model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-1])

    elif model_name == "densenet_model_fold2":
        model = torch.load(f'./models/models/DenseNet/dense_fold2/checkpoint_6.pth.tar', map_location=torch.device('cpu'))['model']
        target_layers = [model.features[-1]]
        model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-1])

    # Random
    elif model_name == "resnet_random_fold0":
        model = torch.load(f'./models/RandomModels/RandomResNet/random_res_fold0/checkpoint_0.pth.tar', map_location=torch.device('cpu'))[
            'model']
        target_layers = [model.layer4[-1]]
        model.fc = torch.nn.Sequential(*list(model.fc.children())[:-1])

    elif model_name == "resnet_random_fold1":
        model = torch.load(f'./models/RandomModels/RandomResNet/random_res_fold1/checkpoint_4.pth.tar', map_location=torch.device('cpu'))[
            'model']
        target_layers = [model.layer4[-1]]
        model.fc = torch.nn.Sequential(*list(model.fc.children())[:-1])

    elif model_name == "resnet_random_fold2":
        model = torch.load(f'./models/RandomModels/RandomResNet/random_res_fold2/checkpoint_5.pth.tar', map_location=torch.device('cpu'))[
            'model']
        target_layers = [model.layer4[-1]]
        model.fc = torch.nn.Sequential(*list(model.fc.children())[:-1])

    elif model_name == "densenet_random_fold0":
        model = \
        torch.load(f'./models/RandomModels/RandomDenseNet/random_dense_fold0/checkpoint_4.pth.tar', map_location=torch.device('cpu'))[
            'model']
        target_layers = [model.features[-1]]
        model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-1])

    elif model_name == "densenet_random_fold1":
        model = \
        torch.load(f'./models/RandomModels/RandomDenseNet/random_dense_fold1/checkpoint_0.pth.tar', map_location=torch.device('cpu'))[
            'model']
        target_layers = [model.features[-1]]
        model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-1])

    elif model_name == "densenet_random_fold2":
        model = \
        torch.load(f'./models/RandomModels/RandomDenseNet/random_dense_fold2/checkpoint_5.pth.tar', map_location=torch.device('cpu'))[
            'model']
        target_layers = [model.features[-1]]
        model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-1])

    else:
        return None, None

    return model, target_layers

# Calculate IOU score
def calculate_IOU(mask, heatmap):
    overlap = mask*heatmap # Logical AND
    union = mask + heatmap # Logical OR
    IOU = overlap.sum()/float(union.sum())
    return IOU

# Calculate dice score
def calculate_dice_score(mask, heatmap, k=1):
    return np.sum(heatmap[mask==k])*2.0 / (np.sum(heatmap) + np.sum(mask))


# To show figure
def save_fig(histogram, bin_edges, fig_name, th=25):
    plt.figure()
    plt.title(f"{fig_name} Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixel count")
    plt.xlim([th, 255])
    plt.plot(bin_edges[th:-(th+1)], histogram[th:-th])
    plt.savefig(f"{fig_name}.jpg")


models = [
    "resnet_special_v2",
    "densenet_special_v2",

    "resnet_model_fold2",
    "densenet_model_fold2",
    "resnet_random_fold2",
    "densenet_random_fold2",

    "resnet_model_fold0",
    "densenet_model_fold0",
    "resnet_random_fold0",
    "densenet_random_fold0",

    "resnet_model_fold1",
    "densenet_model_fold1",
    "resnet_random_fold1",
    "densenet_random_fold1"
]

config = {
    "root_path": "C:/Users/tcemcetin/Documents/YüksekLisans/TrustableAIProject/dataset/rsna-pneumonia-detection-challenge/stage_2_train_images/",
    "img": "00b4ac1b-fa09-4dbe-b93f-7d9e52992a68",
    "aug_smooth": True,
    "eigen_smooth": False,
    "method": "layercam",
    "use_cuda": False,
    "th": 150
}

cnt = 0

root_path = 'C:/Users/tcemcetin/Documents/YüksekLisans/TrustableAIProject/'
df = pd.read_csv(root_path + 'dataset/stage_2_train_labels.csv')
df = df[df.Target == 1]
df['area'] = df.apply(lambda r: r.height * r.width, axis=1)
df = df.sort_values(['patientId', 'area'], ascending=False).drop_duplicates('patientId', keep='first')
df

th = [0, 100, 120, 150, 175, 200]


for model_name in models:
    model, target_layers = get_model(model_name)
    print(f"Model Loaded: {model_name}")

    tmp = df.sample(n=400, random_state=27)

    histogram = []
    cnt = 0
    th_dict = {
        0: [],
        100: [],
        120: [],
        150: [],
        175: [],
        200: []
    }

    for index, row in tmp.iterrows():
        print(f"Index: {cnt} - Patient : {row['patientId']}")
        config['img'] = row['patientId']
        mask, start_point, end_point = create_mask(row)
        cam_image, gb, cam_gb, heatmap, img = gcam(config, model, target_layers)

        # Heatmap oversize
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)
        heatmap = cv2.resize(heatmap, (1024, 1024), interpolation=cv2.INTER_AREA)

        for t in th:
            heatmap_th = (heatmap > t).astype(int)
            IOU = calculate_IOU(mask, heatmap_th)
            th_dict[t].append(IOU)

        #cv2.imshow("aa", cam_image)
        #cv2.imwrite(f"./result_imgs/{model_name}_{cnt}.jpg", cam_image)
        #cv2.waitKey(1)

        cnt += 1
    res = pd.DataFrame.from_dict(th_dict)
    res.to_csv(f"./result/{model_name}.csv")