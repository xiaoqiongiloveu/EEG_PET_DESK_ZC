import asyncio
import csv
import sys
import time
import datetime
from itertools import zip_longest

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, QObject, QEventLoop, QTimer, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QFont
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import qasync
import serial
from serial.tools import list_ports
from ads1299 import ADS1298Data, ADS1298cmd

# 主线程
class MainWindow(QMainWindow, QObject):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.AdsData = ADS1298Data() # 数据缓存及解析
        self.serial_port = None # 串口对象
        self.channel_states = [True] * 8  # 通道显示状态
       #  self.channel_states = [False False False False True True True True]  # 只显示后四个通道
    def initUI(self):

        # 界面的标题
        self.setWindowTitle('ads1299模块-Python示例')

        # 界面的位置和大小的设置
        self.setGeometry(100, 100, 1200, 600)

        # 创建整体界面的水平布局
        hbox = QHBoxLayout()

        # 左侧垂直布局
        left_vbox = QVBoxLayout() #垂直布局
        left_vbox.setAlignment(Qt.AlignTop | Qt.AlignHCenter) #顶部对齐 + 水平居中
        left_vbox.addSpacerItem(QSpacerItem(120, 80, QSizePolicy.Fixed, QSizePolicy.Fixed)) #创建一个固定大小的空白占位区域。
        left_vbox.setSpacing(15) # 相邻控件之间的间距为15像素

        # 创建下拉框/信号类型
        self.port_combo = QComboBox(self) # 创建下拉框，用来显示可用串口
        self.port_combo.addItems([]) # 给下拉框添加一个空列表
        hbox1 = QHBoxLayout() #水平布局
        hbox1.addWidget(QLabel("选择串口")) #显示文字
        hbox1.addWidget(self.port_combo) #显示下拉框

        self.btn_open = QPushButton('打开串口', self)
        self.btn_open.clicked.connect(self.open_serial)
        self.btn_start = QPushButton('开始采集', self)
        self.btn_start.clicked.connect(self.start_collect)
        self.btn_export_csv = QPushButton('导出数据', self)
        self.btn_export_csv.clicked.connect(self.export_csv)

        # 创建通道选择区域 (垂直排列)
        channel_group = QGroupBox("通道选择") #创建分组框
        channel_layout = QVBoxLayout() #垂直布局
        self.channel_buttons = [] #空列表用于保存8个复选框对象
        for i in range(8):
            chk = QCheckBox(f"通道 {i + 1}", self) #创建复选框
            chk.setChecked(True) #复选框默认8个通道全部显示
            chk.toggled.connect(lambda state, idx=i: self.toggle_channel(state, idx))
            channel_layout.addWidget(chk)# 复选框加入垂直布局
            self.channel_buttons.append(chk)
        channel_group.setLayout(channel_layout)#把8通道垂直布局设置为分组框的内部布局。

        left_vbox.addLayout(hbox1)# 把“选择串口标签+下拉框”的水平布局加入左侧垂直布局。
        left_vbox.addWidget(self.btn_open)# 把“打开串口”按钮加入左侧布局。
        left_vbox.addWidget(self.btn_start)# 把“开始采集”按钮加入左侧布局。
        left_vbox.addWidget(self.btn_export_csv)# 把“导出数据”按钮加入左侧布局。
        left_vbox.addWidget(channel_group)# 把通道选择分组框加入左侧布局。

        # 右侧垂直布局
        right_vbox = QVBoxLayout()

        # # 创建一个绘图区域（QWidget）
        self.drawing_area = QWidget(self)#创建固件
        self.drawing_area.setFixedSize(10, 10) #固定部件像素
        left_vbox.addWidget(self.drawing_area)

        # 创建一个FigureCanvas来显示图像
        self.fig, self.axes = plt.subplots(1, 1, layout='constrained')
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        right_vbox.addWidget(self.canvas)# 波形画布加入右侧布局
        right_vbox.addWidget(self.toolbar)# 把工具栏放到画布下方。

        hbox.addLayout(left_vbox)
        hbox.addLayout(right_vbox)

        # 创建一个中心小部件并设置布局
        central_widget = QWidget()
        central_widget.setLayout(hbox)
        self.setCentralWidget(central_widget)

        # 设置支持中文的字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        # 设置图形和坐标轴

        self.lines = []  # 存储8个line对象的列表
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'coral']  # 为每个通道分配不同颜色

        for i in range(8):
            line, = self.axes.plot([], [], color=colors[i], label=f'CH{i + 1}')
            self.lines.append(line) #[] []分别是空的X数据和Y数据。 程序启动曲线存在无数据
        self.axes.grid(True)
        self.axes.legend(loc="upper right")
        self.axes.set_xlim(0, 1500)
        self.axes.set_ylabel("uV")
        self.axes.set_title('八通道脑电')

        # 定时读取数据、处理数据、显示数据
        self.update_show_timer = QTimer(self)
        self.update_show_timer.timeout.connect(self.update_show)
        self.update_show_timer.start(100)

        # 更新串口列表
        self.refresh_ports()

    #  刷新可用的串口列表
    def refresh_ports(self):
        # 保存当前选择的串口
        current_selection = self.port_combo.currentText()
        self.port_combo.clear()

        # 获取所有可用串口
        ports = serial.tools.list_ports.comports()

        if not ports:
            self.port_combo.addItem("未检测到串口")
            self.port_combo.setEnabled(False)
            self.btn_open.setEnabled(False)
        else:
            for port in sorted(ports):
                self.port_combo.addItem(port.device, port.description)
            self.port_combo.setEnabled(True)
            self.btn_open.setEnabled(True)

            # 尝试恢复之前的选择
            if current_selection in [self.port_combo.itemText(i) for i in range(self.port_combo.count())]:
                self.port_combo.setCurrentText(current_selection)
            elif ports:  # 默认选择第一个串口
                self.port_combo.setCurrentIndex(0)

    # 打开或关闭串口
    def open_serial(self):
       if self.btn_open.text() == "打开串口":
            port_name = self.port_combo.currentText()
            # 打开串口，将波特率配置为1000000，数据位为8，停止位为1，无校验位，读超时时间为0.5秒。
            self.serial_port = serial.Serial(port=port_name,
                                             baudrate=1000000,
                                             bytesize=serial.EIGHTBITS,
                                             parity=serial.PARITY_NONE,
                                             stopbits=serial.STOPBITS_ONE,
                                             timeout=0.5)

            if self.serial_port.isOpen():  # 判断串口是否成功打开
                self.btn_open.setText("关闭串口")
                print("打开串口成功。")
                print(self.serial_port.name)  # 输出串口号
            else:
                self.btn_open.setText("打开串口")
                self.serial_port = None
                print("打开串口失败。")
       else:
           self.btn_open.setText("打开串口")
           self.serial_port.write(bytes(ADS1298cmd.stop_collect_cmd()))
           self.serial_port.close()
           self.serial_port = None

    # 开始采集

    def start_collect(self):
        self.serial_port.write(bytes(ADS1298cmd.stop_collect_cmd()))
        if self.btn_start.text() == "开始采集":
            time.sleep(0.5)
            self.btn_start.setText("停止采集")
            self.AdsData.clear()
            if  self.serial_port and self.serial_port.is_open:
                 self.serial_port.write(bytes(ADS1298cmd.start_collect_cmd()))


        else:
            time.sleep(0.5)
            self.btn_start.setText("开始采集")
            if self.serial_port and self.serial_port.is_open:
             self.serial_port.write(bytes(ADS1298cmd.stop_collect_cmd()))


        # 通道切换函数
    def toggle_channel(self, state, channel_idx):
            self.channel_states[channel_idx] = bool(state)
            # 立即更新图形可见性  控制曲线是否可见
            self.lines[channel_idx].set_visible(state)
            # 重新绘制图形
            self.fig.canvas.draw_idle()

    # 更新显示
    def update_show(self):

        if self.serial_port is None:
            return

        # 获取串口已接收的数据
        n = self.serial_port.in_waiting
        if n > 0:
            # 读取串口数据
            com_data = self.serial_port.read(n)
            # 解析数据
            self.AdsData.parse_data(com_data)
            # 更新绘图
            if len(self.AdsData.emg_data) > 0:
                for i in range(8):
                    if self.channel_states[i]:
                        self.update_graph(self.axes, self.lines[i], self.AdsData.emg_data[i], self.AdsData.sample_rate)

                self.fig.canvas.draw()
                self.fig.canvas.flush_events()

    # 更新曲线
    def update_graph(self, axes, line, data, rate):
        if len(data) == 0:
            return

        disp_length = rate * 5
        len_n = len(data)
        if len_n < disp_length:
            disp_length = len_n
        x0 = len_n - disp_length
        x1 = len_n
        channel_index = self.lines.index(line)

        # 固定偏移：每个通道加1000uV
        offset = channel_index * 1000+1000
        y_data = np.array(data[x0:x1]) + offset
        x_data = np.linspace(x0, x1, disp_length)
        line.set_data(x_data, y_data)

        # 设置X轴范围
        axes.set_xlim(x0, x1)

        # 设置Y轴范围
        if any(self.channel_states):
            visible_min = float('inf')
            visible_max = float('-inf')

            for i in range(8):
                if self.channel_states[i] and len(self.AdsData.emg_data[i]) > 0:
                    # 计算偏移后的数据范围
                    offset = i * 1000+1000
                    ch_data = np.array(self.AdsData.emg_data[i][x0:x1]) + offset
                    if ch_data.size > 0:
                        ch_min = np.min(ch_data)
                        ch_max = np.max(ch_data)
                        visible_min = min(visible_min, ch_min)
                        visible_max = max(visible_max, ch_max)

            if visible_min != float('inf') and visible_max != float('-inf'):
                # 计算整体范围并留出边距
                margin = 500
                overall_min = visible_min - margin
                overall_max = visible_max + margin
                # 设置Y轴范围
                axes.set_ylim(overall_min, overall_max)

    # 导出csv数据
    def export_csv(self):

        if len(self.AdsData.emg_data)  == 0 :
            QMessageBox.information(None, "提示", "数据缓存区为空，请先采集数据!", QMessageBox.Ok)
            return

        now = datetime.datetime.now()
        default_name = now.strftime("%Y%m%d_%H%M%S") + "_" + str(self.AdsData.sample_rate) + "sps"
        filepath, _ = QFileDialog.getSaveFileName(
            None,
            "保存CSV文件",
            default_name,
            "CSV文件 (*.csv);;所有文件 (*)"
        )

        if filepath == "":
           return

        # 表头
        headers = [f"通道{i+1}" for i in range(8)]

        # 写入CSV文件（带表头）
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)  # 先写入表头
            # 获取最大数据长度
            max_len = max(len(channel) for channel in self.AdsData.emg_data)

            # 按行写入数据
            for i in range(max_len):
                row = []
                for channel_data in self.AdsData.emg_data:
                    value = channel_data[i] if i < len(channel_data) else ""
                    row.append(f"{value:.3f}" if isinstance(value, float) else "")
                writer.writerow(row)
        QMessageBox.information(None, "提示", "导出成功！", QMessageBox.Ok)

async def main():
    app = QApplication(sys.argv)
    # 使用 qasync 将 asyncio 事件循环与 PyQt 集成
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    # 窗体UI
    main_window = MainWindow()
    main_window.show()

    # 启动事件循环
    with loop:
        loop.run_forever()

if __name__ == '__main__':
    asyncio.run(main())
