import cv2
import pytesseract
import numpy as np
import re
import pyautogui
import winsound

# Set the path to the Tesseract executable if needed (for Windows users)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define the region of interest (ROI) for the screenshot
x, y, w, h = 8, 860, 350, 115

# Capture the screenshot of the defined region
screenshot = pyautogui.screenshot(region=(x, y, w, h))

# Convert the screenshot to a format suitable for OpenCV (numpy array)
screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

# Rescale the image to increase DPI
scale_factor = 4
width = int(screenshot.shape[1] * scale_factor)
height = int(screenshot.shape[0] * scale_factor)
rescaled_image = cv2.resize(screenshot, (width, height), interpolation=cv2.INTER_CUBIC)

# Convert to grayscale
gray = cv2.cvtColor(rescaled_image, cv2.COLOR_BGR2GRAY)

# Binarization
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Noise removal
denoised = cv2.medianBlur(binary, 3)

# Dilation and Erosion
kernel = np.ones((1, 1), np.uint8)
dilated = cv2.dilate(denoised, kernel, iterations=1)
eroded = cv2.erode(dilated, kernel, iterations=1)

# Use Tesseract to perform OCR on the processed image
custom_config = r'--oem 3 --psm 6 -l chi_tra'
detailed_data = pytesseract.image_to_data(eroded, config=custom_config, output_type=pytesseract.Output.DICT)

# Detect the number using regex and locate its position
pattern = re.compile(r'\d+')
number_coords = None
for i, text in enumerate(detailed_data['text']):
    if pattern.match(text):
        number_coords = (detailed_data['left'][i], detailed_data['top'][i], detailed_data['width'][i], detailed_data['height'][i])
        break

# If a number is detected, remove it from the image
if number_coords:
    x, y, w, h = number_coords
    cv2.rectangle(eroded, (x-10, y-10), (x + 30, y + 55), (0, 0, 0), -1)  # Fill the area with black color
    print(f"Number detected at coordinates: {number_coords}")

    # Save intermediate processed image for debugging
    cv2.imwrite("deskewed_no_number.png", eroded)

    # Use Tesseract to perform OCR again on the modified image
    remaining_text = pytesseract.image_to_string(eroded, config=custom_config)
    print("Remaining text after removing number:", remaining_text)

    # Check if "小" is in the remaining text
    if "小" in remaining_text or "j" in remaining_text or "g" in remaining_text:
        # Use regex to find the number before "小" in the original text
        original_text = pytesseract.image_to_string(rescaled_image, config=custom_config)
        match = re.search(r'(\d+)', original_text)
        if match:
            number_before_xiao = match.group(1)
            print(f"Number before '小': {number_before_xiao}")
            winsound.Beep(1000, 500)  # Beep sound
        else:
            print("Number before '小' not found in original text")
    else:
        print("'小' not found in remaining text")

else:
    print("No number detected")

# Display the original screenshot with the ROI outlined (optional)
cv2.rectangle(screenshot, (0, 0), (w, h), (0, 255, 0), 2)
cv2.imshow("Screenshot with ROI", screenshot)
cv2.waitKey(0)
cv2.destroyAllWindows()
