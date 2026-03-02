from scipy.ndimage import binary_dilation
import numpy as np
from PIL import Image

img = Image.open("asia1m/asialakesborder.png").convert("RGB")
arr = np.array(img)

# Define your colors - adjust thresholds to match your actual pixel values
dark_blue = (arr[:,:,0] == 99) & (arr[:,:,1] == 83) & (arr[:,:,2] == 141)   # #63538d
light_blue = (arr[:,:,0] == 150) & (arr[:,:,1] == 200) & (arr[:,:,2] == 250) # #d9f6ff

def color_mask(arr, hex_color, tolerance=0):
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (
        (np.abs(arr[:,:,0].astype(int) - r) <= tolerance) &
        (np.abs(arr[:,:,1].astype(int) - g) <= tolerance) &
        (np.abs(arr[:,:,2].astype(int) - b) <= tolerance)
    )

dark_blue = color_mask(arr, '#63538d', tolerance=25)
light_blue = color_mask(arr, '#96b4f0', tolerance=25)

# Dilate light blue by 1 pixel (8-connected)
light_blue_dilated = binary_dilation(light_blue, structure=np.ones((3,3)))

# Dark blue pixels that touch light blue = keep
# Dark blue pixels that don't touch light blue = erase (set to white)
erase_mask = dark_blue & ~light_blue_dilated

arr[erase_mask] = [255, 255, 255]  # or whatever your background color is

result = Image.fromarray(arr)
result.save("asia1m/asialakesborder2.png")