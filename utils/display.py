from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from pathlib import Path

def plot_bboxes(
  image_file: str,
  bboxes: List[List[float]],
  xywh: bool = True,
  labels: Optional[List[str]] = None
) -> None:
    """
    Args:
      image_file: str specifying the image file path
      bboxes: list of bounding box annotations for all the detections
      xywh: bool, if True, the bounding box annotations are specified as
        [xmin, ymin, width, height]. If False the annotations are specified as
        [xmin, ymin, xmax, ymax]. If you are unsure what the mode is try both
        and check the saved image to see which setting gives the
        correct visualization.

    """
    fig = plt.figure()

    # add axes to the image
    ax = fig.add_axes([0, 0, 1, 1])

    image_folder = Path(image_file).parent

    # read and plot the image
    image = plt.imread(image_file)
    plt.imshow(image)

    # Iterate over all the bounding boxes
    for i, bbox in enumerate(bboxes):
        if xywh:
          xmin, ymin, w, h = bbox
        else:
          xmin, ymin, xmax, ymax = bbox
          w = xmax - xmin
          h = ymax - ymin

        # add bounding boxes to the image
        box = patches.Rectangle(
            (xmin, ymin), w, h, edgecolor="red", facecolor="none"
        )

        ax.add_patch(box)

        if labels is not None:
          rx, ry = box.get_xy()
          cx = rx + box.get_width()/2.0
          cy = ry + box.get_height()/8.0
          l = ax.annotate(
            labels[i],
            (cx, cy),
            fontsize=8,
            fontweight="bold",
            color="white",
            ha='center',
            va='center'
          )
          l.set_bbox(
            dict(facecolor='red', alpha=0.5, edgecolor='red')
          )

    plt.axis('off')
    outfile = os.path.join(image_folder, "image_bbox.png")
    fig.savefig(outfile)

    print("Saved image with detections to %s" % outfile)


