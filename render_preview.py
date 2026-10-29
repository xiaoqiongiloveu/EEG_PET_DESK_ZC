from PIL import Image, ImageDraw, ImageFont
import os

W, H = 1280, 760
img = Image.new('RGB', (W, H), '#071426')
draw = ImageDraw.Draw(img)

# background panel colors
panel = '#0B1D2E'
panel_border = '#2A425E'
button = '#10243B'
muted = '#DDE7F2'
text = '#EAF2FA'

# overall left panel
left_x, top_y, width = 40, 20, 560
panel_rect = (left_x, top_y, left_x + width, H - 30)
draw.rounded_rectangle(panel_rect, radius=18, fill=panel, outline=panel_border, width=2)

# top title bar
# draw a status-like header panel above channel list

def add_button(x, y, w, h, label, color=button):
    draw.rounded_rectangle((x, y, x+w, y+h), radius=10, fill=color, outline='#355574', width=2)
    # simple text
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 26)
        text_w, text_h = draw.textbbox((0, 0), label, font=font)[2:]
        draw.text((x + 18, y + (h - text_h) // 2), label, font=font, fill=text)
    except Exception:
        draw.text((x + 20, y + 10), label, fill=text)

# first row buttons
button_y = 50
button_h = 52
button_w = 220
button_gap = 10
x0 = 70
add_button(x0, button_y, button_w, button_h, '选择串口')
add_button(x0 + button_w + 16, button_y, 300, button_h, '未检测到串口')

add_button(x0, button_y + 70, 500, button_h, '刷新串口')
add_button(x0, button_y + 140, 500, button_h, '打开串口')
add_button(x0, button_y + 210, 500, button_h, '▶ 开始采集')
add_button(x0, button_y + 280, 500, button_h, '■ 停止采集')
add_button(x0, button_y + 350, 500, button_h, '📡 发送 START_STREAM')
add_button(x0, button_y + 420, 500, button_h, '🛑 发送 STOP_STREAM')
add_button(x0, button_y + 490, 500, button_h, '🧪 阻抗测试')

# protocol caption
proto_y = 560
proto_h = 120
outline = '#1E3552'
draw.rounded_rectangle((x0, proto_y, x0 + 500, proto_y + proto_h), radius=10, fill='#091E33', outline=outline, width=2)
try:
    header_font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 24)
    draw.text((x0 + 16, proto_y + 14), 'USB 协议命令', fill='#EAF2FA', font=header_font)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 18)
    draw.text((x0 + 16, proto_y + 54), '状态：待连接', fill='#9EDBFF', font=small)
except Exception:
    draw.text((x0 + 16, proto_y + 14), 'USB 协议命令', fill='#EAF2FA')

# channel drawer style
channel_x = x0
channel_y = 690
channel_w = 500
channel_h = 260
box = (channel_x - 4, channel_y - 4, channel_x + channel_w + 4, channel_y + channel_h + 4)
draw.rounded_rectangle(box, radius=14, fill='#081C2B', outline=panel_border, width=2)

# title row at top of drawer
try:
    title_font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 24)
    draw.text((channel_x + 16, channel_y + 10), '通道选择', fill='#EAF2FA', font=title_font)
except Exception:
    draw.text((channel_x + 16, channel_y + 10), '通道选择', fill='#EAF2FA')

# drawer toggle
try:
    toggle_font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 20)
    draw.rounded_rectangle((channel_x + channel_w - 46, channel_y + 12, channel_x + channel_w - 12, channel_y + 46), radius=8, fill='#10243B', outline='#355574', width=2)
    draw.text((channel_x + channel_w - 38, channel_y + 14), '︿', fill='#EAF2FA', font=toggle_font)
except Exception:
    pass

# channel rows
colors = ['#2F80FF', '#FF7A18', '#F04A9B', '#20C878', '#8D6BEB', '#32B7E9', '#E3A600', '#12B8B5']
row_h = 28
for i, c in enumerate(colors):
    y = channel_y + 52 + i * 30
    draw.rounded_rectangle((channel_x + 10, y, channel_x + channel_w - 10, y + row_h), radius=6, fill=c)
    draw.rounded_rectangle((channel_x + 16, y + 5, channel_x + 28, y + row_h - 5), radius=4, fill=c, outline='rgba(255,255,255,0.85)', width=2)
    # white indicator box
    draw.rounded_rectangle((channel_x + 18, y + 8, channel_x + 29, y + row_h - 8), radius=3, fill='rgba(255,255,255,0.15)', outline='rgba(255,255,255,0.9)', width=2)
    try:
        label_font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 20)
        label = ['Fp1', 'Fp2', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2'][i]
        draw.text((channel_x + 42, y + 2), label, fill='white', font=label_font)
    except Exception:
        draw.text((channel_x + 42, y + 2), ['Fp1', 'Fp2', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2'][i], fill='white')

# right side waveform area
wave_x = 630
wave_y = 30
wave_w = 610
wave_h = 680
wave_bg = '#071426'
draw.rounded_rectangle((wave_x, wave_y, wave_x + wave_w, wave_y + wave_h), radius=12, fill=wave_bg, outline='#1F3851', width=2)

# thin horizontal signal lines
for i in range(8):
    y = wave_y + 30 + i * 80
    draw.line((wave_x + 20, y, wave_x + wave_w - 30, y), fill='#173754', width=1)

# output image
out_path = os.path.join(os.getcwd(), 'ui_effect_preview.png')
img.save(out_path)
print(out_path)
