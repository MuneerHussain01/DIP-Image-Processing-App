import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, 
    QHBoxLayout, QWidget, QFileDialog, QSlider, QStyleFactory, 
    QProgressBar, QCheckBox, QGroupBox, QColorDialog
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QTimer, QPoint


class ImageProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DIP Image Processing App")
        self.setMinimumSize(1000, 600)

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # State
        self.current_image = None
        self.display_image = None
        self.undo_stack = []
        self.redo_stack = []
        self.drawing = False
        self.last_point = QPoint()
        self.theme_dark = False
        self.cap = None
        self.drawing_color = QColor(Qt.red)

        # Image display area
        self.image_label = QLabel("Drag and Drop or Open an Image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed #aaa;")
        self.image_label.setAcceptDrops(True)
        self.image_label.setMinimumSize(640, 480)

        # Progress bar 
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(True)

        # Buttons
        self.btn_open = QPushButton("Open Image")
        self.btn_save = QPushButton("Save Image")
        self.btn_gray = QPushButton("Grayscale")
        self.btn_blur = QPushButton("Gaussian Blur")
        self.btn_edge = QPushButton("Edge Detection")
        self.btn_sharpen = QPushButton("Sharpen")
        self.btn_hue = QPushButton("Hue Adjustment")
        self.btn_reset = QPushButton("Reset")
        self.btn_undo = QPushButton("Undo")
        self.btn_redo = QPushButton("Redo")
        self.btn_live = QPushButton("Start Live Preview")
        self.btn_stop = QPushButton("Stop Live")
        self.btn_theme = QPushButton("Toggle Theme")
        self.btn_draw = QCheckBox("Draw Mode")
        self.btn_color = QPushButton("Change Drawing Color")

        # Sliders
        self.slider_blur = QSlider(Qt.Horizontal)
        self.slider_blur.setMinimum(1)
        self.slider_blur.setMaximum(49)
        self.slider_blur.setValue(5)
        self.slider_blur.setTickInterval(2)

        self.slider_brightness = QSlider(Qt.Horizontal)
        self.slider_brightness.setMinimum(-100)
        self.slider_brightness.setMaximum(100)
        self.slider_brightness.setValue(0)

        self.slider_contrast = QSlider(Qt.Horizontal)
        self.slider_contrast.setMinimum(-100)
        self.slider_contrast.setMaximum(100)
        self.slider_contrast.setValue(0)

        self.slider_threshold1 = QSlider(Qt.Horizontal)
        self.slider_threshold1.setMinimum(0)
        self.slider_threshold1.setMaximum(255)
        self.slider_threshold1.setValue(50)

        self.slider_threshold2 = QSlider(Qt.Horizontal)
        self.slider_threshold2.setMinimum(0)
        self.slider_threshold2.setMaximum(255)
        self.slider_threshold2.setValue(150)

        self.slider_hue = QSlider(Qt.Horizontal)
        self.slider_hue.setMinimum(-180)
        self.slider_hue.setMaximum(180)
        self.slider_hue.setValue(0)

        # Timer for live cam
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_live)

        # Layouts
        main_layout = QHBoxLayout(central_widget)
        control_layout = QVBoxLayout()

        # Group boxes
        basic_group = QGroupBox("Basic Operations")
        basic_layout = QVBoxLayout()
        for w in [self.btn_open, self.btn_save, self.btn_gray, self.btn_reset, self.btn_undo, self.btn_redo]:
            basic_layout.addWidget(w)
        basic_group.setLayout(basic_layout)

        filter_group = QGroupBox("Filters")
        filter_layout = QVBoxLayout()
        for w in [self.btn_blur, self.slider_blur, self.btn_edge, self.slider_threshold1, 
                 self.slider_threshold2, self.btn_sharpen, self.btn_hue, self.slider_hue]:
            filter_layout.addWidget(w)
        filter_group.setLayout(filter_layout)

        adjust_group = QGroupBox("Adjustments")
        adjust_layout = QVBoxLayout()
        for w in [self.slider_brightness, self.slider_contrast]:
            adjust_layout.addWidget(w)
        adjust_group.setLayout(adjust_layout)

        live_group = QGroupBox("Live/Drawing")
        live_layout = QVBoxLayout()
        for w in [self.btn_live, self.btn_stop, self.btn_draw, self.btn_color]:
            live_layout.addWidget(w)
        live_group.setLayout(live_layout)

        # Add groups to control layout
        control_layout.addWidget(basic_group)
        control_layout.addWidget(filter_group)
        control_layout.addWidget(adjust_group)
        control_layout.addWidget(live_group)
        control_layout.addWidget(self.btn_theme)
        control_layout.addStretch()
        control_layout.addWidget(self.progress)

        # Control panel
        control_panel = QWidget()
        control_panel.setLayout(control_layout)
        control_panel.setFixedWidth(350)

        # Main layout
        main_layout.addWidget(self.image_label, 70)  # 70% width
        main_layout.addWidget(control_panel, 30)    # 30% width

        # Connect actions
        self.connect_actions()

        # Initial state
        self.btn_stop.setEnabled(False)

    def connect_actions(self):
        """Connect all signals to slots"""
        self.btn_open.clicked.connect(self.open_image)
        self.btn_save.clicked.connect(self.save_image)
        self.btn_gray.clicked.connect(self.convert_gray)
        self.btn_blur.clicked.connect(self.apply_blur)
        self.btn_edge.clicked.connect(self.detect_edges)
        self.btn_sharpen.clicked.connect(self.sharpen_image)
        self.btn_hue.clicked.connect(self.adjust_hue)
        self.btn_reset.clicked.connect(self.reset_image)
        self.btn_undo.clicked.connect(self.undo)
        self.btn_redo.clicked.connect(self.redo)
        self.btn_theme.clicked.connect(self.toggle_theme)
        self.btn_live.clicked.connect(self.start_live)
        self.btn_stop.clicked.connect(self.stop_live)
        self.btn_draw.stateChanged.connect(self.toggle_draw)
        self.btn_color.clicked.connect(self.change_drawing_color)

        self.slider_blur.valueChanged.connect(self.apply_blur)
        self.slider_brightness.valueChanged.connect(self.adjust_brightness_contrast)
        self.slider_contrast.valueChanged.connect(self.adjust_brightness_contrast)
        self.slider_threshold1.valueChanged.connect(self.detect_edges)
        self.slider_threshold2.valueChanged.connect(self.detect_edges)
        self.slider_hue.valueChanged.connect(self.adjust_hue)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        path = event.mimeData().urls()[0].toLocalFile()
        self.load_image(path)

    def mousePressEvent(self, event):
        if self.drawing and event.buttons() == Qt.LeftButton and self.image_label.pixmap():
            self.last_point = event.pos() - self.image_label.pos()

    def mouseMoveEvent(self, event):
        if self.drawing and event.buttons() == Qt.LeftButton and self.image_label.pixmap():
            painter = QPainter(self.image_label.pixmap())
            pen = QPen(self.drawing_color, 5, Qt.SolidLine)
            painter.setPen(pen)
            current_point = event.pos() - self.image_label.pos()
            painter.drawLine(self.last_point, current_point)
            self.last_point = current_point
            self.image_label.update()

    def toggle_draw(self, state):
        self.drawing = bool(state)

    def change_drawing_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.drawing_color = color

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.load_image(path)

    def load_image(self, path):
        self.progress.setValue(30)
        self.current_image = cv2.imread(path)
        self.progress.setValue(70)
        if self.current_image is not None:
            self.undo_stack = [self.current_image.copy()]
            self.redo_stack.clear()
            self.display_image_on_label(self.current_image)
        self.progress.setValue(100)

    def convert_gray(self):
        if self.display_image is not None:
            gray = cv2.cvtColor(self.display_image, cv2.COLOR_BGR2GRAY)
            result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            self.push_history(result)
            self.display_image_on_label(result)

    def apply_blur(self):
        if self.display_image is not None:
            k = self.slider_blur.value()
            if k % 2 == 0: k += 1
            blur = cv2.GaussianBlur(self.display_image, (k, k), 0)
            self.push_history(blur)
            self.display_image_on_label(blur)

    def detect_edges(self):
        if self.display_image is not None:
            gray = cv2.cvtColor(self.display_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, self.slider_threshold1.value(), self.slider_threshold2.value())
            result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            self.push_history(result)
            self.display_image_on_label(result)

    def sharpen_image(self):
        if self.display_image is not None:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(self.display_image, -1, kernel)
            self.push_history(sharpened)
            self.display_image_on_label(sharpened)

    def adjust_hue(self):
        if self.display_image is not None:
            hsv = cv2.cvtColor(self.display_image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            h = (h + self.slider_hue.value()) % 180
            hsv = cv2.merge([h, s, v])
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            self.push_history(result)
            self.display_image_on_label(result)

    def adjust_brightness_contrast(self):
        if self.current_image is not None:
            brightness = self.slider_brightness.value()
            contrast = self.slider_contrast.value()

            img = self.current_image.astype(np.int16)
            img = img * (1 + contrast / 100.0) + brightness
            img = np.clip(img, 0, 255).astype(np.uint8)

            self.display_image_on_label(img)

    def reset_image(self):
        if self.undo_stack:
            self.display_image_on_label(self.undo_stack[0])
            self.undo_stack = [self.undo_stack[0]]
            self.redo_stack.clear()

    def save_image(self):
        if self.display_image is not None:
            path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png);;JPG Files (*.jpg)")
            if path:
                self.progress.setValue(50)
                cv2.imwrite(path, self.display_image)
                self.progress.setValue(100)

    def start_live(self):
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.btn_live.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.timer.start(30)
        else:
            self.btn_live.setEnabled(True)

    def stop_live(self):
        if self.cap:
            self.timer.stop()
            self.cap.release()
            self.cap = None
        self.btn_live.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def update_live(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.display_image_on_label(frame)

    def undo(self):
        if len(self.undo_stack) > 1:
            self.redo_stack.append(self.undo_stack.pop())
            self.display_image_on_label(self.undo_stack[-1])

    def redo(self):
        if self.redo_stack:
            img = self.redo_stack.pop()
            self.undo_stack.append(img)
            self.display_image_on_label(img)

    def toggle_theme(self):
        if self.theme_dark:
            QApplication.setStyle(QStyleFactory.create("Fusion"))
            self.setStyleSheet("")
        else:
            dark = """
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QPushButton {
                background-color: #444;
                color: white;
                border: 1px solid #555;
                padding: 5px;
            }
            QGroupBox {
                border: 1px solid #555;
                border-radius: 5px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #555;
            }
            QSlider::handle:horizontal {
                width: 18px;
                margin: -5px 0;
                background: #ddd;
            }
            """
            self.setStyleSheet(dark)
        self.theme_dark = not self.theme_dark

    def display_image_on_label(self, img):
        self.display_image = img
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))

    def push_history(self, img):
        self.undo_stack.append(img.copy())
        self.redo_stack.clear()
        self.current_image = img


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = ImageProcessor()
    win.show()
    sys.exit(app.exec_())