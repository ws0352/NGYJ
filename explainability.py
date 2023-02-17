from pytorch_grad_cam import AblationCAM, EigenCAM
from pytorch_grad_cam.ablation_layer import AblationLayerFasterRCNN
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_accross_batch_and_channels, scale_cam_image

targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]
target_layers = CondConv(in_planes=32, out_planes=64, kernel_size=3, stride=1, padding=1, bias=False)

# or a very fast alternative:

cam = EigenCAM(model,
              target_layers,
              use_cuda=torch.nn.cuda.is_available(),
              reshape_transform=fasterrcnn_reshape_transform)
