import os
import struct
import zlib

W, H = 1280, 760
img = [[(0, 0, 0, 255) for _ in range(W)] for _ in range(H)]

BG = (7, 20, 38, 255)
PANEL = (11, 29, 46, 255)
PANEL_BORDER = (42, 66, 94, 255)
BUTTON = (16, 36, 59, 255)
BORDER = (52, 88, 122, 255)
TEXT = (234, 242, 250, 255)
PLOT_BG = (7, 20, 38, 255)
GRID = (23, 53, 82, 255)

COLORS = [
    (47, 128, 255, 255),
    (255, 122, 24, 255),
    (240, 74, 155, 255),
    (32, 200, 120, 255),
    (141, 107, 235, 255),
    (50, 183, 233, 255),
    (227, 166, 0, 255),
    (18, 184, 181, 255),
]


def set_pixel(x, y, color):
    if 0 <= x < W and 0 <= y < H:
        img[y][x] = color


def fill_rect(x0, y0, x1, y1, color):
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    for y in range(y0, y1):
        for x in range(x0, x1):
            set_pixel(x, y, color)


def draw_round_rect(x0, y0, x1, y1, r, color, outline=None, outline_width=1):
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    r = min(r, (x1 - x0) // 2, (y1 - y0) // 2)
    if r <= 0:
        fill_rect(x0, y0, x1, y1, color)
        if outline:
            fill_rect(x0, y0, x1, y0 + outline_width, outline)
            fill_rect(x0, y1 - outline_width, x1, y1, outline)
            fill_rect(x0, y0, x0 + outline_width, y1, outline)
            fill_rect(x1 - outline_width, y0, x1, y1, outline)
        return
    # fill body
    fill_rect(x0 + r, y0, x1 - r, y1, color)
    fill_rect(x0, y0 + r, x1, y1 - r, color)
    for y in range(y0 + r, y1 - r + 1):
        for x in range(x0, x1 + 1):
            set_pixel(x, y, color)
    # corners
    for cy in range(r):
        for cx in range(r):
            if (cx - r + 1) ** 2 + (cy - r + 1) ** 2 <= r * r:
                for dx, dy in [(cx, cy), (x1 - x0 - cx, cy), (cx, y1 - y0 - cy), (x1 - x0 - cx, y1 - y0 - cy)]:
                    x = x0 + dx
                    y = y0 + dy
                    set_pixel(x, y, color)
    if outline:
        # top and bottom
        for y in range(y0, y0 + outline_width):
            for x in range(x0, x1):
                set_pixel(x, y, outline)
        for y in range(y1 - outline_width, y1):
            for x in range(x0, x1):
                set_pixel(x, y, outline)
        # left and right
        for x in range(x0, x0 + outline_width):
            for y in range(y0, y1):
                set_pixel(x, y, outline)
        for x in range(x1 - outline_width, x1):
            for y in range(y0, y1):
                set_pixel(x, y, outline)


def draw_line(x0, y0, x1, y1, color, width=1):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        for ox in range(-width // 2, width // 2 + 1):
            for oy in range(-width // 2, width // 2 + 1):
                set_pixel(x0 + ox, y0 + oy, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def draw_text(x, y, text, color, size=22):
    # Very simple text using block pixels; purpose is preview only.
    font = [
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+-:()" 
    ]
    # Build a basic 5x7 font set for common glyphs.
    glyphs = {
        ' ': [0,0,0,0,0],
        'A': [0,1,1,1,0, 1,0,0,0,1, 1,1,1,1,1, 1,0,0,0,1, 1,0,0,0,1],
        'B': [1,1,1,1,0, 1,0,0,0,1, 1,1,1,1,0, 1,0,0,0,1, 1,1,1,1,0],
        'C': [0,1,1,1,1, 1,0,0,0,0, 1,0,0,0,0, 1,0,0,0,0, 0,1,1,1,1],
        'D': [1,1,1,1,0, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,1,1,1,0],
        'E': [1,1,1,1,1, 1,0,0,0,0, 1,1,1,1,0, 1,0,0,0,0, 1,1,1,1,1],
        'F': [1,1,1,1,1, 1,0,0,0,0, 1,1,1,1,0, 1,0,0,0,0, 1,0,0,0,0],
        'G': [0,1,1,1,1, 1,0,0,0,0, 1,0,1,1,1, 1,0,0,0,1, 0,1,1,1,1],
        'H': [1,0,0,0,1, 1,0,0,0,1, 1,1,1,1,1, 1,0,0,0,1, 1,0,0,0,1],
        'I': [1,1,1,1,1, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 1,1,1,1,1],
        'J': [0,0,0,1,0, 0,0,1,0,0, 0,0,1,0,0, 1,0,1,0,0, 0,1,0,0,0],
        'K': [1,0,0,0,1, 1,0,0,1,0, 1,0,1,0,0, 1,1,0,0,0, 1,0,0,0,1],
        'L': [1,0,0,0,0, 1,0,0,0,0, 1,0,0,0,0, 1,0,0,0,0, 1,1,1,1,1],
        'M': [1,0,0,0,1, 1,1,0,1,1, 1,0,1,0,1, 1,0,0,0,1, 1,0,0,0,1],
        'N': [1,0,0,0,1, 1,1,0,0,1, 1,0,1,0,1, 1,0,0,1,1, 1,0,0,0,1],
        'O': [0,1,1,1,0, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 0,1,1,1,0],
        'P': [1,1,1,1,0, 1,0,0,0,1, 1,1,1,1,0, 1,0,0,0,0, 1,0,0,0,0],
        'Q': [0,1,1,1,0, 1,0,0,0,1, 1,0,0,0,1, 1,0,1,0,1, 0,1,1,1,1],
        'R': [1,1,1,1,0, 1,0,0,0,1, 1,1,1,1,0, 1,0,1,0,0, 1,0,0,1,1],
        'S': [0,1,1,1,1, 1,0,0,0,0, 0,1,1,1,0, 0,0,0,0,1, 1,1,1,1,0],
        'T': [1,1,1,1,1, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0],
        'U': [1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 0,1,1,1,0],
        'V': [1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 0,1,0,1,0, 0,0,1,0,0],
        'W': [1,0,0,0,1, 1,0,0,0,1, 1,0,1,0,1, 1,1,0,1,1, 0,1,0,1,0],
        'X': [1,0,0,0,1, 0,1,0,1,0, 0,0,1,0,0, 0,1,0,1,0, 1,0,0,0,1],
        'Y': [1,0,0,0,1, 0,1,0,1,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0],
        'Z': [1,1,1,1,1, 0,0,0,1,0, 0,0,1,0,0, 0,1,0,0,0, 1,1,1,1,1],
        '0': [0,1,1,1,0, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 0,1,1,1,0],
        '1': [0,0,1,0,0, 0,1,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,1,1,1,0],
        '2': [0,1,1,1,0, 1,0,0,0,1, 0,0,0,1,0, 0,0,1,0,0, 1,1,1,1,1],
        '3': [1,1,1,1,1, 0,0,0,0,1, 0,1,1,1,0, 0,0,0,0,1, 1,1,1,1,0],
        '4': [0,0,0,1,0, 0,0,1,1,0, 0,1,0,1,0, 1,1,1,1,1, 0,0,0,1,0],
        '5': [1,1,1,1,1, 1,0,0,0,0, 1,1,1,1,0, 0,0,0,0,1, 1,1,1,1,0],
        '6': [0,1,1,1,0, 1,0,0,0,0, 1,1,1,1,0, 1,0,0,0,1, 0,1,1,1,0],
        '7': [1,1,1,1,1, 0,0,0,0,1, 0,0,0,1,0, 0,0,1,0,0, 0,0,1,0,0],
        '8': [0,1,1,1,0, 1,0,0,0,1, 0,1,1,1,0, 1,0,0,0,1, 0,1,1,1,0],
        '9': [0,1,1,1,0, 1,0,0,0,1, 0,1,1,1,1, 0,0,0,0,1, 0,1,1,1,0],
        '+': [0,0,1,0,0, 0,0,1,0,0, 1,1,1,1,1, 0,0,1,0,0, 0,0,1,0,0],
        '-': [0,0,0,0,0, 0,0,0,0,0, 1,1,1,1,1, 0,0,0,0,0, 0,0,0,0,0],
        ':': [0,0,0,0,0, 0,0,1,0,0, 0,0,0,0,0, 0,0,1,0,0, 0,0,0,0,0],
        '.': [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,1,0,0, 0,0,0,0,0],
        '(': [0,0,1,0,0, 0,1,0,0,0, 1,0,0,0,0, 0,1,0,0,0, 0,0,1,0,0],
        ')': [0,0,1,0,0, 0,0,0,1,0, 0,0,0,0,1, 0,0,0,1,0, 0,0,1,0,0],
        '/': [0,0,0,0,1, 0,0,0,1,0, 0,0,1,0,0, 0,1,0,0,0, 1,0,0,0,0],
        '▣': [1,1,1,1,1, 1,0,0,0,1, 1,0,1,0,1, 1,0,0,0,1, 1,1,1,1,1],
        '▶': [0,0,0,1,0, 0,0,1,1,0, 0,1,1,1,0, 1,1,1,0,0, 0,1,1,0,0],
        '■': [1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1],
        '╳': [1,0,0,0,1, 0,1,0,1,0, 0,0,1,0,0, 0,1,0,1,0, 1,0,0,0,1],
        '▭': [1,1,1,1,1, 1,0,0,0,1, 1,0,1,0,1, 1,0,0,0,1, 1,1,1,1,1],
    }
    px = x
    for ch in text:
        g = glyphs.get(ch, glyphs[' '])
        for i in range(25):
            if g[i]:
                xx = px + (i % 5)
                yy = y + (i // 5)
                for dx in range(size//2):
                    for dy in range(size//2):
                        set_pixel(xx * (size//2) + dx, yy * (size//2) + dy, color)
        px += 6 * (size//2)

# fill main background
fill_rect(0, 0, W, H, BG)

# left panel
left_x, top_y, left_w = 30, 20, 570
draw_round_rect(left_x, top_y, left_x + left_w, H - 25, 16, PANEL, outline=PANEL_BORDER, outline_width=2)

# buttons across left area
btn_y = 44
btn_h = 52
for idx, label in enumerate(['选择串口', '未检测到串口', '刷新串口', '打开串口', '▶ 开始采集', '■ 停止采集', '📡 发送 START_STREAM', '🛑 发送 STOP_STREAM', '🧪 阻抗测试']):
    if idx == 1:
        draw_round_rect(left_x + 220, btn_y + idx * 60, left_x + 220 + 300, btn_y + idx * 60 + btn_h, 10, BUTTON, outline=BORDER, outline_width=2)
    else:
        draw_round_rect(left_x + 25, btn_y + idx * 60, left_x + 25 + 500, btn_y + idx * 60 + btn_h, 10, BUTTON, outline=BORDER, outline_width=2)
    if idx == 0:
        draw_text(left_x + 45, btn_y + idx * 60 + 12, '选择串口', TEXT, size=26)
    elif idx == 1:
        draw_text(left_x + 245, btn_y + idx * 60 + 12, '未检测到串口', TEXT, size=26)
    elif idx == 2:
        draw_text(left_x + 45, btn_y + idx * 60 + 12, '刷新串口', TEXT, size=26)
    elif idx == 3:
        draw_text(left_x + 45, btn_y + idx * 60 + 12, '打开串口', TEXT, size=26)
    elif idx == 4:
        draw_text(left_x + 45, btn_y + idx * 60 + 12, '▶ 开始采集', TEXT, size=26)
    elif idx == 5:
        draw_text(left_x + 45, btn_y + idx * 60 + 12, '■ 停止采集', TEXT, size=26)
    elif idx == 6:
        draw_text(left_x + 45, btn_y + idx * 60 + 12, '📡 发送 START_STREAM', TEXT, size=26)
    elif idx == 7:
        draw_text(left_x + 45, btn_y + idx * 60 + 12, '🛑 发送 STOP_STREAM', TEXT, size=26)
    elif idx == 8:
        draw_text(left_x + 45, btn_y + idx * 60 + 12, '🧪 阻抗测试', TEXT, size=26)

# protocol panel
proto_y = 560
proto_h = 100
draw_round_rect(left_x + 25, proto_y, left_x + 25 + 500, proto_y + proto_h, 10, (9, 30, 51, 255), outline=(30, 53, 82, 255), outline_width=2)
draw_text(left_x + 50, proto_y + 12, 'USB 协议命令', TEXT, size=24)
draw_text(left_x + 50, proto_y + 52, '状态：待连接', (158, 219, 255, 255), size=20)

# channel panel drawer header
channel_x = left_x + 25
channel_y = 677
channel_w = 500
channel_h = 240
draw_round_rect(channel_x - 4, channel_y - 4, channel_x + channel_w + 4, channel_y + channel_h + 4, 16, (8, 28, 43, 255), outline=PANEL_BORDER, outline_width=2)
draw_text(channel_x + 16, channel_y + 10, '通道选择', TEXT, size=24)
# toggle button
xbtn = channel_x + channel_w - 46
ybtn = channel_y + 11
# draw arrow as a tiny symbol
for yy in range(ybtn, ybtn + 22):
    for xx in range(xbtn, xbtn + 26):
        if xx >= xbtn and xx <= xbtn + 20 and yy >= ybtn and yy <= ybtn + 20:
            fill_rect(xx, yy, xx + 1, yy + 1, (16, 36, 59, 255))

draw_round_rect(xbtn, ybtn, xbtn + 28, ybtn + 28, 8, BUTTON, outline=BORDER, outline_width=2)
draw_text(xbtn + 7, ybtn + 4, '︿', TEXT, size=18)

# rows
for i, c in enumerate(COLORS):
    y = channel_y + 52 + i * 28
    draw_round_rect(channel_x + 12, y, channel_x + channel_w - 12, y + 24, 6, c)
    draw_round_rect(channel_x + 18, y + 5, channel_x + 30, y + 19, 3, (255, 255, 255, 120), outline=(255, 255, 255, 200), outline_width=1)
    draw_text(channel_x + 42, y + 2, ['Fp1', 'Fp2', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2'][i], (255, 255, 255, 255), size=18)

# right plot area
plot_x, plot_y, plot_w, plot_h = 640, 30, 600, 700
draw_round_rect(plot_x, plot_y, plot_x + plot_w, plot_y + plot_h, 12, PLOT_BG, outline=(31, 56, 81, 255), outline_width=2)
for i in range(8):
    yy = plot_y + 40 + i * 80
    draw_line(plot_x + 20, yy, plot_x + plot_w - 20, yy, GRID, width=1)
    draw_line(plot_x + 20, yy + 32, plot_x + plot_w - 20, yy + 32, (27, 72, 110, 255), width=1)

# write PNG
raw = bytearray()
for y in range(H):
    raw.append(0)
    row = img[y]
    for x in range(W):
        raw.extend(row[x])

def png_chunk(tag, data):
    return struct.pack('!I', len(data)) + tag + data + struct.pack('!I', zlib.crc32(tag + data) & 0xffffffff)

png = b'\x89PNG\r\n\x1a\n'
png += png_chunk(b'IHDR', struct.pack('!IIBBBBB', W, H, 8, 6, 0, 0, 0))
png += png_chunk(b'IDAT', zlib.compress(bytes(raw), 9))
png += png_chunk(b'IEND', b'')
with open(os.path.join(os.getcwd(), 'ui_effect_preview.png'), 'wb') as f:
    f.write(png)
print(os.path.join(os.getcwd(), 'ui_effect_preview.png'))
