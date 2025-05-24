import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

known_faces = {
    "Name1": "Image1.jpg",
    "Name2": "Image2.jpg"
}

known_descriptors = {}

for name, img_file in known_faces.items():
    image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: {img_file} not found!")
        continue
    kp, des = orb.detectAndCompute(image, None)
    if des is not None:
        known_descriptors[name] = des

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    kp2, des2 = orb.detectAndCompute(gray, None)

    for (x, y, w, h) in faces:
        name = "Unknown"

        if des2 is not None:
            best_match = None
            max_matches = 0

            for person, des1 in known_descriptors.items():
                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)
                if len(matches) > max_matches and len(matches) > 30:
                    max_matches = len(matches)
                    best_match = person

            if best_match:
                name = best_match

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255 if name != "Unknown" else 0, 0 if name != "Unknown" else 255), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255 if name != "Unknown" else 0, 0 if name != "Unknown" else 255), 2)

    cv2.imshow('Multi-Person Face Detection', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
