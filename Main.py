# Khai báo thư viện sử dụng YOLO v8 để nhận diện xe và tính toán tốc độ
from ultralytics import YOLO
from ultralytics.solutions import speed_estimation

# Khai báo sử dụng thư viện OpenCV
import cv2

# Nạp model Yolo đã được huấn luyện trước
model = YOLO("yolov8n.pt")
names = model.model.names

# Đọc file video
cap = cv2.VideoCapture("test1.mp4")
assert cap.isOpened(), "Lỗi đọc file"

# Lấy giá trị chiều rộng, cao và số khung hình của video lưu bào biến để xử lý sau này
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Lưu lại video có chứa các tham số nhận diện tốc độ
video_writer = cv2.VideoWriter("test_speed_estimation.avi",
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps,
                               (w, h))

line_pts = [(0, 360), (1280, 360)]

# Khởi tạo đối tượng speed_obj để ước tính tốc độ
speed_obj = speed_estimation.SpeedEstimator()
speed_obj.set_args(reg_pts=line_pts,
                   names=names,
                   view_img=True)

#Lặp qua khung hình, xử lý vết và đo tốc độ
while cap.isOpened():

    success, im0 = cap.read()
    if not success:
        break

    tracks = model.track(im0, persist=True, show=False)

    im0 = speed_obj.estimate_speed(im0, tracks)
    video_writer.write(im0)

# Giải phóng đối tượng lưu video, đóng và giải phóng các cửa sổ ứng dụng
cap.release()
video_writer.release()
cv2.destroyAllWindows()