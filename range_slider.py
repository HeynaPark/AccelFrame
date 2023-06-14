from swipe_speed import *
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QSlider, QHBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt, pyqtSignal, QObject


class SliderValueSender(QObject):
    slider_value_changed = pyqtSignal(int, int)


class RangeSlider(QWidget):
    def __init__(self, min_value, max_value):
        super().__init__()

        self.init_ui(min_value, max_value)
        self.size = 0

    def init_ui(self, min_value, max_value):
        self.size = max_value - min_value
        size = self.size
        center_frame = min_value + int((size)*0.5)
        print(size, center_frame)
        layout = QHBoxLayout()

        # 최소값 레이블 생성
        self.min_label = QLabel(f"{min_value}: {min_value + int(size*0.2)}")
        layout.addWidget(self.min_label)
        # 최소값 슬라이더 생성
        self.min_slider = QSlider()
        self.min_slider.setOrientation(Qt.Horizontal)
        self.min_slider.setMinimum(min_value)
        self.min_slider.setMaximum(center_frame)
        self.min_slider.setTickPosition(QSlider.TicksBothSides)  # 양쪽에 눈금 표시
        self.min_slider.setTickInterval(10)  # 눈금 간격
        self.min_slider.setValue(min_value + int(size*0.2))
        self.min_slider.setStyleSheet("""
            QSlider {
                background-color: transparent;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #CCCCCC;
                margin: 2px 0;
            }
            QSlider::sub-page:horizontal {
                background: #66CCFF;
                border-radius: 4px;
            }
            QSlider::add-page:horizontal {
                background: #CCCCCC;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #66CCFF;
                border: 1px solid #999999;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                background: #55AAFF;
            }
        """)

        layout.addWidget(self.min_slider)

        # 최대값 슬라이더 생성
        self.max_slider = QSlider()
        self.max_slider.setOrientation(Qt.Horizontal)
        self.max_slider.setMinimum(center_frame)
        self.max_slider.setMaximum(max_value)
        self.max_slider.setTickPosition(QSlider.TicksBothSides)  # 양쪽에 눈금 표시
        self.max_slider.setTickInterval(10)  # 눈금 간격
        self.max_slider.setValue(max_value - int(size*0.2))
        self.max_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #CCCCCC;
                margin: 2px 0;
            }
            QSlider::sub-page:horizontal {
                background: #CCCCCC;
                border-radius: 4px;
            }
            QSlider::add-page:horizontal {
                background: #66CCFF;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #66CCFF;
                border: 1px solid #999999;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
        """)
        layout.addWidget(self.max_slider)

        # 최대값 레이블 생성
        self.max_label = QLabel(f"{max_value - int(size*0.2)}: {max_value}")
        layout.addWidget(self.max_label)

        self.button = QPushButton("make")  # 버튼 생성
        layout.addWidget(self.button)  # 버튼을 레이아웃에 추가

        # 슬라이더 값 변경 시 호출되는 슬롯 설정
        self.min_slider.valueChanged.connect(self.update_min_label)
        self.max_slider.valueChanged.connect(self.update_max_label)

        self.setLayout(layout)

        self.sender = SliderValueSender()
        self.button.clicked.connect(self.send_slider_value)

    def send_slider_value(self):
        set_front_value(self.min_slider.value())
        set_back_value(self.max_slider.value())
        make()

    def update_min_label(self, value):
        self.min_label.setText(f"{min_value}: {value}")

    def update_max_label(self, value):
        self.max_label.setText(f"{value}: {max_value}")


min_value = get_min_value()
max_value = get_max_value()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    slider_window = RangeSlider(min_value, max_value)
    slider_window.resize(1000, 200)  # 위젯 사이즈 조정
    slider_window.show()
    sys.exit(app.exec_())
