import cv2

# Load image
image_path = "IMG_3521.jpeg"
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load Haar cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:
    # Draw green circle around face
    center_x, center_y = x + w // 2, y + h // 2
    radius = int(0.5 * (w + h) // 2)
    cv2.circle(img, (center_x, center_y), radius, (0, 255, 0), 3)

    # Region of interest for eyes
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)

    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)

# Add text
cv2.putText(img, "this is me", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Save the result
output_path = "output_image.jpg"
cv2.imwrite(output_path, img)
print(f"Image saved to {output_path}")
