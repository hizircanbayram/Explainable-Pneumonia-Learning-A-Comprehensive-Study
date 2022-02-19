import cv2
import numpy as np
import torch
from torchvision import models
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image


def get_model(model_name):
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

    
    if model_name == "resnet_50":
        model = models.resnet50(pretrained=True)
        target_layers = [model.layer4[-1]]
    
    elif model_name == "densenet161":
        model = models.densenet161(pretrained=True)
        target_layers = [model.features[-1]]

    else:
        return None, None

    return model, target_layers


def gcam(args, model, target_layers):


    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = None

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    cam_algorithm = args.method
    with cam_algorithm(model=model,
                        target_layers=target_layers,
                        use_cuda=args.use_cuda) as cam:

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32

        grayscale_cam = cam(input_tensor=input_tensor,
                            target_category=target_category,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    gb = gb_model(input_tensor, target_category=target_category)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    return cam_image, gb, cam_gb


class methods:
    gradcam = GradCAM
    scorecam = ScoreCAM
    gradcamplus = GradCAMPlusPlus
    ablationcam = AblationCAM
    xgradcam = XGradCAM
    eigencam = EigenCAM
    eigengradcam = EigenGradCAM
    layercam = LayerCAM
    fullgrad = FullGrad


class Arguments:
    image_path = "./1.jpg"
    aug_smooth = True
    eigen_smooth = False
    method = methods.gradcam
    use_cuda = False
   
    def print_text( self ):
        print(self.str)

if __name__ == "__main__":
    #cam_image, gb, cam_gb = gcam(Arguments(), "resnet_50")
    model_name = "densenet161"
    model, target_layers = get_model(model_name)
    print("Model Loaded")

    cam_image, gb, cam_gb = gcam(Arguments(), model, target_layers)
    plt.imshow(cam_image)
