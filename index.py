import cv2
import pytesseract
import numpy as np
import re
import winsound
import pyautogui

# Set the path to the Tesseract executable if needed (for Windows users)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define the region of interest (ROI) for the screenshot
# (x, y, width, height) - Modify these values as needed
x, y, w, h = 8, 860, 350, 115

# Capture the screenshot of the defined region
screenshot = pyautogui.screenshot(region=(x, y, w, h))

# Convert the screenshot to a format suitable for OpenCV (numpy array)
screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

# Resize the image to enlarge the text
scale_percent = 1000  # Increase the scale percentage to enlarge the image more
width = int(screenshot.shape[1] * scale_percent / 100)
height = int(screenshot.shape[0] * scale_percent / 100)
dim = (width, height)
resized_screenshot = cv2.resize(screenshot, dim, interpolation=cv2.INTER_LINEAR)

# Convert the resized screenshot to grayscale
gray_roi = cv2.cvtColor(resized_screenshot, cv2.COLOR_BGR2GRAY)

# Apply a Gaussian blur to reduce noise
gray_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)

# Apply a sharpening filter
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
gray_roi = cv2.filter2D(src=gray_roi, ddepth=-1, kernel=kernel)

# Apply thresholding
gray_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# Save intermediate processed images for debugging
cv2.imwrite("gray_roi.png", gray_roi)

# Use Tesseract to perform OCR on the cropped and processed ROI
# Specify the Chinese language ('chi_sim' for Simplified Chinese, 'chi_tra' for Traditional Chinese)
custom_config = r'--oem 3 --psm 4 -l chi_tra'
text = pytesseract.image_to_string(gray_roi, config=custom_config)

print("Detected text:", text)

# Use regular expression to find the number before "小财前" or "小財前"
match = re.search(r'(\d+)\s*小', text)

if match:
    number = match.group(1)
    print("Number before '小财前':", number)
    winsound.Beep(1000, 500)
else:
    print("Pattern not found")

# Display the original screenshot with the ROI outlined (optional)
cv2.rectangle(screenshot, (0, 0), (w, h), (0, 255, 0), 2)
cv2.imshow("Screenshot with ROI", screenshot)
cv2.waitKey(0)
cv2.destroyAllWindows()
