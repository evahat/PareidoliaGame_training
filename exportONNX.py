import torch
from PareidoliaGame.training import TinyYOLOv2

#--------------------------------------------------------------------
# This script loads your trained PyTorch model and exports it to ONNX format.
# Make sure to adjust the MODEL_PATH and ONNX_PATH before running.
#--------------------------------------------------------------------

MODEL_PATH = "model_emotion_bg_final.pth"   # your .pth file
ONNX_PATH = "emotion_yolo.onnx"
IMG_SIZE = 416

model = TinyYOLOv2()
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(checkpoint)

model.eval()  # IMPORTANT

dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["X"],
    output_names=["Y"]
    # dynamic_axes={
    #     "input": {0: "batch_size"},
    #     "output": {0: "batch_size"}
    # }
)

print("Model exported to", ONNX_PATH)
