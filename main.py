from simple_facerec import SimpleFacerec
import cv2

# Inicializa reconhecimento facial
sfr = SimpleFacerec()
sfr.load_encoding_images(r"C:\Users\Cassiano Alvariza\Desktop\facerec")

# Inicia webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # largura
cap.set(4, 480)  # altura

process_this_frame = True
face_locations = []
face_names = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Reduz o tamanho do frame para acelerar
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    if process_this_frame:
        # Detecta rostos no frame menor
        face_locations, face_names = sfr.detect_known_faces(small_frame)

        # Corrige as coordenadas para o tamanho original
        face_locations = [(t*4, r*4, b*4, l*4) for (t, r, b, l) in face_locations]

    process_this_frame = not process_this_frame  # alterna entre processar e pular

    # Desenha caixas e nomes
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 200), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)

    cv2.imshow("Reconhecimento Facial", frame)

    # Pressiona ESC para sair
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()