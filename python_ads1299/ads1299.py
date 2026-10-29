# ==========================================================================
# 文件总览 —— 上位机侧 ADS1299/ADS1298 串口协议层
# --------------------------------------------------------------------------
# 本模块是 PC 端（上位机）与 EEG 采集设备通信的协议实现，与固件侧
# binary_protocol.h / binary_protocol.c 严格对应，主要做三件事：
#
#   1. 组包下发：ADS1298cmd 提供 start_stream / stop_stream / set_config /
#      impedance 等命令构造，按 BP 帧格式封装后经串口发给设备。
#   2. 流式解包：ADS1298Data.parse_data() 接收串口字节流，内部维护累积缓冲
#      rx_buff，自动处理“粘包 / 断包 / 噪声丢弃”，边收边解。
#   3. 数据解码：把原始字节还原为 8 通道脑电（uV）、电极阻抗、
#      加速度/陀螺仪、电池电压；EEG 微伏值原样保存供显示滤波和导出。
#
# 兼容设计：同时支持三种帧格式，按优先级依次判定
#   ① BP 二进制帧（55 AA 开头，带 CRC16 校验）—— 现行固件使用
#   ② 遗留 ADS 帧（A5 或 AA 开头，带异或校验）—— 老固件
#   ③ ASCII 文本行（以 CR/LF 结尾）—— 老固件的文本响应
#
# 关键约定
#   - 所有多字节字段均为小端（little-endian），与固件保持一致
#   - 三个 _parse_* 方法统一返回三态：
#       True  = 已消费或丢弃若干字节，外层循环应继续解析
#       False = 数据尚不完整，等待下一批串口数据
#       None  = 不属于本格式，交给下一个分支尝试
#   - CRC / 帧长校验失败时只丢弃 1 个字节而不是清空缓冲，
#     相当于滑动窗口重同步，可避免把负载里恰好出现的帧头误丢
# ==========================================================================
import struct                                                 # 标准库：二进制打包/解包

# ===== 协议常量：必须与固件 binary_protocol.h 中的定义逐一对齐 =====
BP_SYNC = b"\x55\xAA"                                         # 帧同步头，固件端同样使用 55 AA
BP_TYPE_EEG = 0x01                                            # EEG 采样数据帧
BP_TYPE_IMP = 0x03                                            # 电极阻抗测试结果帧
BP_TYPE_STATUS = 0x04                                         # 设备状态查询应答帧
BP_TYPE_AUX = 0x05                                            # 辅助数据帧（IMU + 电量）
BP_TYPE_CMD = 0x10                                            # 上位机下发的命令帧
BP_TYPE_CMD_RESP = 0x11                                       # 设备对命令的应答帧
BP_CMD_START_STREAM = 0x01                                    # 开始采集
BP_CMD_STOP_STREAM = 0x02                                     # 停止采集
BP_CMD_SET_CONF = 0x03                                        # 配置采集参数
BP_CMD_TEST_IMP = 0x0E                                        # 阻抗测试
BP_CMD_STOP_IMP = 0x14                                        # 停止阻抗测试
BP_TYPE_IMU = 0x02                                             # IMU 独立帧
BP_CMD_GET_STATE = 0x04                                        # 查询设备状态
BP_CMD_ENABLE_DEVICE = 0x0B                                    # 使能设备
BP_CMD_DISABLE_DEVICE = 0x0C                                   # 关闭设备
BP_CMD_CLEAR_DATA = 0x11                                       # 清空设备端缓存
BP_CMD_PAUSE = 0x12                                            # 暂停采集
BP_CMD_RESUME = 0x13                                           # 恢复采集
LINK_STATS_MAGIC = b"ND"                                       # STATUS 扩展块标识
LINK_STATS_SIZE = 14                                           # 标识和三个 uint32 计数器
 # 协议缺口提示：固件侧还实现了下列帧类型 / 命令，本模块尚未定义常量，
 # _handle_binary_frame 中也没有对应分支：
 #   0x20 文件 / OTA 分块上传帧      —— GET_FILE 时使用
 #   0x7E DIAG 帧                    —— 固件 1Hz 主动上报的运行诊断
 #   BP_CMD_GET_STATE                —— 查询状态，固件回 BP_TYPE_STATUS 帧
 #   BP_CMD_LIST_FILES / DELETE_FILE / CLEAR_DATA
 #   BP_CMD_DISABLE_DEVICE / ENABLE_DEVICE
 # 这些帧能通过 CRC 与长度校验，但因无匹配分支会被静默丢弃（仅存入 last_binary_frame）。
 # 若后续要做文件回传或状态轮询，需在此补常量并在解码分支中接上。

# 固件上传的是 ADC 原始计数值，乘该系数换算为微伏(uV)。
# 与固件 main.c 中 Send_EEG_Packet() 的换算公式完全相同：
#   Vref = 4.5V，PGA 增益 = 24，24 位有符号补码满量程 2^23-1
ADC_COUNT_TO_UV = 4500000.0 / 8388607.0 / 24.0                # ADC 计数值 -> uV 的换算系数


# CRC16-CCITT（初值 0xFFFF，多项式 0x1021），逐字节计算，无反射、无末尾异或。
# 注意：只覆盖 body（不含同步头、长度域与 CRC 域本身）。
def bp_crc16(data):
    """计算 CRC16-CCITT，与固件 binary_protocol.c 实现一致（初值 0xFFFF）。"""
    crc = 0xFFFF                                              # CRC 初值 0xFFFF
    for byte in data:                                         # 逐字节参与计算
        crc ^= byte << 8                                                # 当前字节移入高 8 位并异或
        for _ in range(8):                                              # 每字节左移 8 次
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF  # 最高位为 1 则异或多项式 0x1021
    return crc                                                # 返回 16 位校验值


# BP 帧结构：55 AA | len(2B,小端) | body | crc(2B,小端)
#   body = 帧类型(1B) + 序号(2B) + 时间戳(4B) + 负载数据
def bp_pack(frame_type, seq, payload=b"", timestamp=0):
    """构造 BP 帧：55 AA + 长度 + body + 小端 CRC16。"""
    payload = bytes(payload)                                  # 统一转成 bytes，便于后续拼接
    body = struct.pack("<BHI", frame_type, seq & 0xFFFF, timestamp & 0xFFFFFFFF) + payload  # 拼接 type + seq + timestamp + payload
    if len(body) > 512 or len(body) < 7:                      # body 长度必须在 7~512 之间
        raise ValueError("BP payload must produce a body length between 7 and 512 bytes")  # 长度非法时直接抛异常
    return BP_SYNC + struct.pack("<H", len(body)) + body + struct.pack("<H", bp_crc16(body))  # 同步头 + 长度 + body + CRC


# 上位机数据解码主类：一个实例对应一路串口连接，内部持有接收缓冲。
class ADS1298Data:
    """同时支持老固件 ADS 帧与现行 BP 二进制协议的解码器。"""

    def __init__(self):
        self.rx_buff = bytearray()                                      # 累积接收缓冲（跨批次保留未解析完的字节）
        self.emg_data = [[] for _ in range(8)]                          # 8 通道原始微伏数据，每通道一个 list
        self.new_raw_samples = []                                       # 最近一次 parse_data 解出的滤波前 8 通道 uV 样本
        self.last_ascii_response = ""                                   # 最近一次 ASCII 文本响应
        self.last_binary_frame = None                                   # 最近一帧解析结果（调试用）
        self.impedance_data = None                                      # 8 通道阻抗值，单位由固件决定
        self.acceleration = None                                        # 三轴加速度，单位 g
        self.gyroscope = None                                           # 三轴角速度，单位 dps
        self.battery_voltage = None                                     # 电池电压，单位 V
        self.motion_state = None                                        # 运动状态标志（由固件定义）
        self.sample_rate = 250                                          # 当前采样率，默认 250Hz
        self.reset_link_counters()

    def reset_link_counters(self):
        """重置当前连接的链路计数，不清空已采样本。"""
        self.frames_received = 0
        self.eeg_frames_received = 0
        self.crc_errors = 0
        self.malformed_frames = 0
        self.packet_counts = {}
        self.eeg_frame_events = []
        self.device_status = None
        self.last_status_seq = None

    # 清空一次采集会话的全部状态（保留采样率）。
    def clear(self):
        self.rx_buff.clear()                                            # 清空接收缓冲
        self.emg_data = [[] for _ in range(8)]                          # 清空 8 通道已采集数据
        self.new_raw_samples = []                                       # 清空尚未交给实时检测的样本
        self.last_binary_frame = None                                   # 清空最近帧记录
        self.impedance_data = None                                      # 清空阻抗数据
        self.acceleration = None                                        # 清空加速度
        self.gyroscope = None                                           # 清空陀螺仪
        self.battery_voltage = None                                     # 清空电池电压
        self.motion_state = None                                        # 清空运动状态

    # 记录老固件返回的 ASCII 文本行（如 OK / ERR 提示）。
    def _handle_ascii_response(self, line):
        self.last_ascii_response = line                                 # 仅保存，不做解析

    # 协议层只保存原始微伏值，显示滤波由界面独立执行。
    def _append_channels(self, channels):
        if len(channels) >= 8:
            sample = tuple(float(value) for value in channels[:8])
            self.new_raw_samples.append(sample)
            for channel, value in enumerate(sample):
                self.emg_data[channel].append(value)

    # 按帧类型分发处理。各分支的 payload 长度阈值即为该类型的最小合法长度。
    # 未识别的帧类型不会抛异常，只存入 last_binary_frame 后被静默丢弃。
    def _handle_binary_frame(self, frame_type, seq, timestamp, payload):
        self.last_binary_frame = (frame_type, seq, timestamp, bytes(payload))  # 原样保存整帧，便于调试或二次解析
        if frame_type == BP_TYPE_EEG:
            self.eeg_frames_received += 1
            self.eeg_frame_events.append((seq, timestamp))

        if frame_type == BP_TYPE_EEG and len(payload) >= 35:            # EEG 帧：至少 3(status) + 32(8×int32)
            # 负载布局：status[3] + 8 个 int32 ADC 值[32] + IMU 字段[12]
            raw_values = struct.unpack_from("<8i", payload, 3)          # 从偏移 3 开始解 8 个 int32（跳过 3 字节状态字）
            channels_uv = [value * ADC_COUNT_TO_UV for value in raw_values]  # 换算为 uV
            self._append_channels(channels_uv)                          # 保存原始微伏数据
            if len(payload) >= 47:                                      # 负载满 47 字节说明还带 12 字节 IMU 数据
                self._update_motion_fields(payload[35:47], include_battery=False)  # 解析尾部运动数据（EEG 帧不含电量）
        elif frame_type == BP_TYPE_IMP and len(payload) >= 16:          # 阻抗帧：8 个 uint16
            self.impedance_data = list(struct.unpack_from("<8H", payload, 0))  # 解出 8 通道阻抗
        elif frame_type == BP_TYPE_AUX and len(payload) >= 16:          # AUX 帧：至少 16 字节
            self._update_motion_fields(payload, include_battery=True)   # 解析运动数据（AUX 帧含电量）
        elif frame_type == BP_TYPE_IMU and len(payload) >= 16:
            self._update_motion_fields(payload, include_battery=True)
        elif frame_type == BP_TYPE_STATUS:
            self.device_status = self._parse_status_payload(payload)
            self.last_status_seq = seq
        elif frame_type == BP_TYPE_CMD_RESP and len(payload) >= 2:      # 命令应答帧：至少 2 字节（命令 ID + 状态）
            cmd_id, status = payload[0], payload[1]                     # 取出命令 ID 与状态码
            message = payload[2:].split(b"\0", 1)[0].decode("utf-8", "replace")  # 消息体到第一个 \0 结束，容错解码
            prefix = "OK" if status == 0 else "ERR"                     # 状态码 0 为 OK，其余为 ERR
            self.last_ascii_response = f"{prefix}:CMD_{cmd_id:02X} {message}".strip()  # 统一格式化为可读文本

    def _parse_status_payload(self, payload):
        """解析 STATUS 帧，兼容没有 ND 扩展块的旧固件。"""
        status = {
            "device_type": None, "sd_usage_percent": None, "sd_ok": None,
            "battery_v": None, "state_code": None, "task_id": None,
            "pb_status": None, "firmware": "", "serial_no": "",
            "ads_drops": 0, "wireless_tx_overflow": 0,
            "wireless_rx_overflow": 0, "link_stats": False,
            "raw": bytes(payload),
        }
        if len(payload) >= 11:
            status.update(
                device_type=payload[0], sd_usage_percent=payload[1],
                sd_ok=bool(payload[2]),
                battery_v=int.from_bytes(payload[3:5], "little") / 100.0,
                state_code=payload[5],
                task_id=int.from_bytes(payload[6:10], "little"),
                pb_status=payload[10],
            )
            blob = payload[11:]
            ver_raw, _, rest = blob.partition(b"\x00")
            sn_raw, _, trailer = rest.partition(b"\x00")
            status["firmware"] = ver_raw.decode("ascii", errors="replace")
            status["serial_no"] = sn_raw.decode("ascii", errors="replace")
            if len(trailer) >= LINK_STATS_SIZE and trailer[:2] == LINK_STATS_MAGIC:
                status["ads_drops"] = int.from_bytes(trailer[2:6], "little")
                status["wireless_tx_overflow"] = int.from_bytes(trailer[6:10], "little")
                status["wireless_rx_overflow"] = int.from_bytes(trailer[10:14], "little")
                status["link_stats"] = True
        elif len(payload) >= 5:
            status["sd_ok"] = bool(payload[2])
            status["battery_v"] = int.from_bytes(payload[3:5], "little") / 1000.0
        return status

    # 解析 IMU 与电量字段。EEG 帧尾部只带 12 字节运动数据（无电量），
    # AUX 帧则带完整 16 字节（含电量与运动状态），故按 include_battery 区分格式串。
    def _update_motion_fields(self, payload, include_battery):
        """解析固件 AUX / EEG 帧中的运动字段，不影响已有的 EEG 数据。"""
        if include_battery:                                             # AUX 帧：6 个 int16 + 电量 uint16 + 状态 uint8
            values = struct.unpack_from("<6hHBB", payload, 0)           # 解出 6 个运动分量 + 电量 + 状态
            self.acceleration = tuple(value / 1000.0 for value in values[:3])  # 加速度：原始值 /1000 得 g
            self.gyroscope = tuple(value / 10.0 for value in values[3:6])  # 陀螺仪：原始值 /10 得 dps
            self.battery_voltage = values[6] / 100.0                    # 电量：原始值 /100 得 V
            self.motion_state = values[7]                               # 运动状态标志位
        else:                                                           # EEG 帧尾部：只有 6 个 int16
            values = struct.unpack_from("<6h", payload, 0)              # 解出 6 个运动分量
            self.acceleration = tuple(value / 1000.0 for value in values[:3])  # 加速度换算
            self.gyroscope = tuple(value / 10.0 for value in values[3:6])  # 陀螺仪换算

    # 解析一帧 BP 二进制数据。返回值含义见 parse_data 中的说明。
    def _parse_binary_frame(self):
        if len(self.rx_buff) < 4:                                       # 连同步头和长度域都不够，等待更多数据
            return False

        body_len = struct.unpack_from("<H", self.rx_buff, 2)[0]         # 取出 body 长度（小端 uint16）
        if body_len < 7 or body_len > 512:                              # 长度非法说明同步失败
            self.malformed_frames += 1
            del self.rx_buff[0]                                         # 丢弃 1 字节后重新同步
            return True                                                 # 已处理，外层继续循环

        total_len = 4 + body_len + 2                                    # 整帧长度 = 同步头2 + 长度域2 + body + CRC2
        if len(self.rx_buff) < total_len:                               # 数据未收全，等待下一批
            return False

        frame = self.rx_buff[:total_len]                                # 截取完整一帧
        body = frame[4:4 + body_len]                                    # 取出 body 部分
        received_crc = struct.unpack_from("<H", frame, 4 + body_len)[0]  # 取出帧尾携带的 CRC
        if bp_crc16(body) != received_crc:                              # 校验失败
            self.crc_errors += 1
            del self.rx_buff[0]                                         # 只丢 1 字节，滑动重同步
            return True                                                 # 继续循环

        frame_type, seq, timestamp = struct.unpack_from("<BHI", body, 0)  # 从 body 头部解出 type / seq / timestamp
        self.frames_received += 1
        self.packet_counts[frame_type] = self.packet_counts.get(frame_type, 0) + 1
        self._handle_binary_frame(frame_type, seq, timestamp, body[7:])  # body[7:] 即真正的 payload
        del self.rx_buff[:total_len]                                    # 消费掉这一整帧
        return True                                                     # 解析成功

    # 解析老固件的 ADS 帧：A5 开头为数据/参数帧，AA 开头为命令回执帧。
    def _parse_legacy_frame(self):
        if self.rx_buff[0] == 0xA5:                                     # A5：老固件的数据 / 参数帧
            if len(self.rx_buff) < 4:                                   # 至少需要 4 字节才能校验
                return False
            if self.rx_buff[3] != (self.rx_buff[1] ^ self.rx_buff[2]):  # 第 4 字节为前两字节的异或校验
                del self.rx_buff[0]                                     # 校验失败
                return True
            frame_len = self.rx_buff[1] + 3                             # 整帧长度 = 长度域 + 3
            if frame_len > len(self.rx_buff):                           # 未收全，继续等待
                return False
            if self.rx_buff[frame_len - 1] != 0x5A:                     # 帧尾必须是 0x5A
                del self.rx_buff[0]                                     # 帧尾不对，丢弃首字节重同步
                return True
            self.frame_unpack(bytes(self.rx_buff[:frame_len]))          # 交给 frame_unpack 按地址解析
            del self.rx_buff[:frame_len]                                # 消费掉这一帧
            return True

        if self.rx_buff[0] == 0xAA:                                     # AA：老固件的命令回执帧
            if len(self.rx_buff) < 2:                                   # 至少需要 2 字节
                return False
            frame_len = self.rx_buff[1] + 2                             # 整帧长度 = 长度域 + 2
            if frame_len > len(self.rx_buff):                           # 未收全，继续等待
                return False
            if self.rx_buff[frame_len - 1] != 0xBB:                     # 帧尾必须是 0xBB
                del self.rx_buff[0]                                     # 丢弃首字节重同步
                return True
            del self.rx_buff[:frame_len]                                # 回执帧无有效数据，直接丢弃
            return True                                                 # 继续循环

        return None                                                     # 既不是 A5 也不是 AA，交给 ASCII 分支处理

    # 串口数据入口：把新到字节追加进累积缓冲，然后循环尝试解析出完整帧。
    # 循环的每一轮都要么消费掉若干字节、要么因数据不足而 break，不会原地空转。
    def parse_data(self, data_in):
        self.new_raw_samples = []                                       # 每次只交付本批完整帧中的新样本
        self.eeg_frame_events = []
        if data_in is None:                                             # 空输入直接返回
            return
        self.rx_buff.extend(data_in if isinstance(data_in, (bytes, bytearray)) else bytes(data_in))  # 兼容 bytes / bytearray / 可迭代整数序列

        while self.rx_buff:                                             # 缓冲非空就持续尝试解析
            if self.rx_buff.startswith(BP_SYNC):                        # 优先判定现行二进制协议
                if not self._parse_binary_frame():                      # 返回 False 表示数据不足
                    break                                               # 退出循环等待更多数据
                continue                                                # 本帧已处理，进入下一轮

            legacy_result = self._parse_legacy_frame()                  # 再尝试老固件帧格式
            if legacy_result is False:                                  # 数据不足，退出等待
                break
            if legacy_result is True:                                   # 已处理，继续
                continue

            # ASCII 分支只用于老固件的文本响应。二进制负载已在前面的分支处理完毕，
            # 因此这里即使遇到负载中的 CR/LF 字节也不会被误判成文本行。
            first = self.rx_buff[0]                                     # 取首字节判断是否为可打印 ASCII
            if first in (9, 10, 13) or 0x20 <= first <= 0x7E:           # TAB / LF / CR 或 0x20~0x7E 视为文本行起点
                line_end = next(                                        # 查找行结束符
                    (idx for idx, byte in enumerate(self.rx_buff) if byte in (0x0A, 0x0D)),
                    None,
                )
                if line_end is None:                                    # 未找到换行，等待更多数据
                    break
                line = self.rx_buff[:line_end].decode("utf-8", errors="replace").strip()  # 按 UTF-8 容错解码并去空白
                del self.rx_buff[:line_end + 1]                         # 连同换行符一起消费
                if line:                                                # 空行忽略
                    self._handle_ascii_response(line)                   # 记录该文本响应
                continue                                                # 继续循环

            del self.rx_buff[0]                                         # 不可识别的噪声字节，丢弃后继续寻找帧头

    # 解老固件的 A5 数据帧：data[2] 为寄存器地址，data[4] 起为有效载荷。
    def frame_unpack(self, data):
        frame_type = data[2]                                            # 第 3 字节为寄存器地址
        if frame_type == ADS1298cmd.ADDRESS_SAMPLE_PAR:                 # 地址 = 采样率参数
            sample_rates = [125, 250, 500, 1000, 2000]                  # 固件支持的采样率档位表
            if data[4] < len(sample_rates):                             # 索引越界保护
                self.sample_rate = sample_rates[data[4]]                # 按档位设置采样率
        elif frame_type == ADS1298cmd.ADDRESS_START:                    # 地址 = 开始采集（数据帧）
            data_num = int((data[1] - 2) / (4 * 8))                     # 每个样本 8 个 float32 = 32 字节，取整得样本数
            values = struct.unpack(f"<{8 * data_num}f", data[4:4 + data_num * 32])  # 一次性解出全部样本
            for offset in range(data_num):                              # 逐样本处理
                self._append_channels(values[offset * 8:(offset + 1) * 8])  # 取该样本的 8 个通道值


# 命令构造集合：每个方法返回可直接写入串口的完整字节串。
class ADS1298cmd:
    # 以下为老固件保留的寄存器地址常量，供旧调用方兼容使用
    ADDRESS_HW_VERSION = 0x01                                 # 硬件版本
    ADDRESS_SOFTWARE = 0x02                                   # 软件版本
    ADDRESS_DEVICE_NAME = 0x03                                # 设备名称
    ADDRESS_DEVICE_MAC = 0x04                                 # 设备 MAC
    ADDRESS_POWER = 0x05                                      # 电源控制
    ADDRESS_RESET = 0x06                                      # 复位
    ADDRESS_SAMPLE_PAR = 0x10                                 # 采样率参数（读回时用于同步 sample_rate）
    ADDRESS_START = 0x11                                      # 采集数据帧

    # 通用命令封装：命令 ID 占 1 字节，其后紧跟参数，整体作为 BP 帧负载。
    @staticmethod
    def binary_command(command_id, args=b"", seq=0):
        return bp_pack(BP_TYPE_CMD, seq, bytes([command_id]) + bytes(args))  # 命令 ID 拼在参数之前，seq 默认 0

    # 采集参数配置命令（BP_CMD_SET_CONF）。参数顺序与固件解析端必须一致。
    @staticmethod
    def set_config_cmd(sample_rate=250, gain=24, channel_mask=0xFF,
                       input_type=0, bias_enabled=1, test_signal=0,
                       duration_sec=0):
        args = struct.pack(                                             # 参数打包：<HBBBBBI 共 13 字节
            "<HBBBBBI",   # 格式串：采样率H 增益B 通道掩码B 输入类型B 偏置B 测试信号B 时长I
            int(sample_rate), int(gain), int(channel_mask), int(input_type),  # 采样率(2) 增益(1) 通道掩码(1) 输入类型(1)
            int(bias_enabled), int(test_signal), int(duration_sec),     # 偏置使能(1) 测试信号(1) 采集时长(4)
        )
        return ADS1298cmd.binary_command(BP_CMD_SET_CONF, args)         # 封装为 SET_CONF 命令帧

    # 开始采集。
    @staticmethod
    def start_stream_cmd():
        return ADS1298cmd.binary_command(BP_CMD_START_STREAM)           # 无参数命令

    # 停止采集。
    @staticmethod
    def stop_stream_cmd():
        return ADS1298cmd.binary_command(BP_CMD_STOP_STREAM)            # 无参数命令

    # 全通道电极阻抗测试。
    @staticmethod
    def impedance_cmd():
        return ADS1298cmd.binary_command(BP_CMD_TEST_IMP)               # 无参数命令

    @staticmethod
    def stop_impedance_cmd():
        return ADS1298cmd.binary_command(BP_CMD_STOP_IMP)

    @staticmethod
    def enable_device_cmd():
        return ADS1298cmd.binary_command(BP_CMD_ENABLE_DEVICE)

    @staticmethod
    def disable_device_cmd():
        return ADS1298cmd.binary_command(BP_CMD_DISABLE_DEVICE)

    @staticmethod
    def get_state_cmd():
        return ADS1298cmd.binary_command(BP_CMD_GET_STATE)

    @staticmethod
    def pause_cmd():
        return ADS1298cmd.binary_command(BP_CMD_PAUSE)

    @staticmethod
    def resume_cmd():
        return ADS1298cmd.binary_command(BP_CMD_RESUME)

    # 早期按钮代码使用的是这两个旧方法名。它们现已改为指向现行固件协议，
    # 而不再是已废弃的 AA 帧实现。
    start_collect_cmd = start_stream_cmd                      # 旧名 -> 开始采集
    stop_collect_cmd = stop_stream_cmd                        # 旧名 -> 停止采集

    # 构造老固件的 AA 命令帧（已过时，仅供兼容旧设备）：
    #   AA | len | 0x80写/0x81读 | addr | data... | xor | BB
    @staticmethod
    def legacy_cmd_data_pack(addr, is_write, data):
        cmd = [0xAA, len(data) + 3, 0x80 if is_write else 0x81, addr]   # AA | 长度 | 读写位 | 地址
        cmd.extend(data)                                                # 追加数据域
        cmd.extend([0x00, 0xBB])                                        # 预留校验位并以 0xBB 结尾
        xor = 0                                                         # 异或校验初值
        for byte in cmd[1:-1]:                                          # 对长度域到数据域逐字节异或
            xor ^= byte                                                 # 累加异或
        cmd[-2] = xor                                                   # 回填校验字节
        return cmd                                                      # 返回完整命令字节列表

