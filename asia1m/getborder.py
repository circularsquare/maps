from scipy.ndimage import binary_dilation
import numpy as np
from PIL import Image

img = Image.open("asia1m/asialakesne.png").convert("RGB")
arr = np.array(img)

white_mask = np.mean(arr, axis=2) > 240

dilated_white = binary_dilation(white_mask, structure=np.ones((3,3)))

blue_mask = np.mean(arr, axis=2) < 240  # dark/saturated pixels as "blue"
border_mask = dilated_white & blue_mask

print("white pixels:", white_mask.sum())
print("blue pixels:", blue_mask.sum())
print("border pixels:", border_mask.sum())

arr[border_mask] = [99, 83, 141]

result = Image.fromarray(arr)
result.save("asia1m/asialakesborder.png")