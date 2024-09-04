import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import cv2
import torch
from torchvision.transforms import Compose
from run_monodepth import DPTDepthModel, MidasNet_large
from run_monodepth import NormalizeImage, Resize, PrepareForNet

def run_webcam(model_path, model_type="dpt_hybrid", optimize=True):
    """Run MonoDepthNN on webcam feed to compute depth maps."""
    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("device: %s" % device)

    # load network
    if model_type == "dpt_large":  # DPT-Large
        net_w = net_h = 384
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid":  # DPT-Hybrid
        net_w = net_h = 384
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "midas_v21":  # Convolutional model
        net_w = net_h = 384
        model = MidasNet_large(model_path, non_negative=True)
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        assert False, f"model_type '{model_type}' not implemented."

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    model.eval()

    if optimize and device == torch.device("cuda"):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()

    model.to(device)

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("start processing webcam feed")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_input = transform({"image": img})["image"]

        # Compute depth map
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)

            if optimize and device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()

            prediction = model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        # Normalize the prediction to display as image
        depth_min = prediction.min()
        depth_max = prediction.max()
        prediction = (prediction - depth_min) / (depth_max - depth_min)
        prediction = (255 * prediction).astype("uint8")

        # Display result
        cv2.imshow("Depth Map", prediction)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("finished")


if __name__ == "__main__":
    # Set your model weights path and type here
    model_weights = "weights/dpt_hybrid-midas-501f0c75.pt"
    model_type = "dpt_hybrid"

    # Run depth estimation on webcam feed
    run_webcam(model_weights, model_type, optimize=True)
