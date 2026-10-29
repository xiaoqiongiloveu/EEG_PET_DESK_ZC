from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


def load_font(size):
    candidates = [
        r'C:\\Windows\\Fonts\\msyh.ttc',
        r'C:\\Windows\\Fonts\\simhei.ttf',
        r'C:\\Windows\\Fonts\\msjhl.ttc',
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def create_pet_icon(path: str):
    width = height = 512
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    bg = (255, 145, 0, 255)
    draw.rounded_rectangle((0, 0, width, height), radius=84, fill=bg)

    # cat face and dog face
    cat_center = (180, 250)
    dog_center = (334, 250)

    def draw_face(center, scale, fur_color, ear_color, accent_color):
        x, y = center
        draw.ellipse((x - 120 * scale, y - 110 * scale, x + 120 * scale, y + 110 * scale), fill=fur_color)
        # ears
        draw.polygon([(x - 50 * scale, y - 88 * scale), (x - 15 * scale, y - 160 * scale), (x + 15 * scale, y - 88 * scale)], fill=ear_color)
        draw.polygon([(x + 35 * scale, y - 88 * scale), (x + 70 * scale, y - 160 * scale), (x + 100 * scale, y - 88 * scale)], fill=ear_color)
        # eyes
        draw.ellipse((x - 48 * scale, y - 20 * scale, x - 18 * scale, y + 10 * scale), fill=(55, 45, 35, 255))
        draw.ellipse((x + 18 * scale, y - 20 * scale, x + 48 * scale, y + 10 * scale), fill=(55, 45, 35, 255))
        # nose / cheeks
        draw.ellipse((x - 16 * scale, y + 12 * scale, x + 16 * scale, y + 42 * scale), fill=(240, 170, 160, 255))
        draw.arc((x - 35 * scale, y + 28 * scale, x + 35 * scale, y + 85 * scale), 200, 340, fill=(55, 45, 35, 255), width=5)
        # mouth
        draw.arc((x - 52 * scale, y + 66 * scale, x + 52 * scale, y + 118 * scale), 205, 335, fill=(55, 45, 35, 255), width=5)
        # cheeks
        draw.ellipse((x - 70 * scale, y + 18 * scale, x - 48 * scale, y + 40 * scale), fill=(255, 180, 170, 255))
        draw.ellipse((x + 48 * scale, y + 18 * scale, x + 70 * scale, y + 40 * scale), fill=(255, 180, 170, 255))
        # paw pads / muzzle for dog
        if accent_color == (110, 66, 38, 255):
            draw.ellipse((x - 20 * scale, y + 80 * scale, x + 20 * scale, y + 120 * scale), fill=(255, 255, 255, 255))

    # cat face
    draw_face(cat_center, 1.0, (255, 255, 255, 255), (255, 255, 255, 255), (255, 180, 140, 255))
    # dog face
    draw_face(dog_center, 1.0, (255, 255, 255, 255), (255, 255, 255, 255), (110, 66, 38, 255))
    # dog dark patch
    draw.ellipse((dog_center[0] + 28, dog_center[1] - 22, dog_center[0] + 110, dog_center[1] + 60), fill=(118, 83, 52, 255))

    # top paw icon
    paw_x, paw_y = 256, 72
    draw.ellipse((paw_x - 58, paw_y - 18, paw_x + 58, paw_y + 42), fill=(255, 255, 255, 255), outline=(75, 46, 15, 255), width=8)
    paw = [(paw_x - 18, paw_y - 6), (paw_x - 36, paw_y + 32), (paw_x - 12, paw_y + 54), (paw_x + 12, paw_y + 54), (paw_x + 36, paw_y + 32), (paw_x + 18, paw_y - 6)]
    draw.polygon(paw, fill=(255, 255, 255, 255), outline=(75, 46, 15, 255), width=8)
    draw.ellipse((paw_x - 10, paw_y - 18, paw_x + 10, paw_y + 14), fill=(255, 153, 0, 255))

    # paws on bottom
    for px in (120, 176, 336, 392):
        draw.ellipse((px - 48, 370, px + 48, 450), fill=(255, 255, 255, 255), outline=(75, 46, 15, 255), width=8)

    # whiskers and extra details
    for dx in (-72, -48):
        draw.line((cat_center[0] - 72, cat_center[1] + 18, cat_center[0] - 120, cat_center[1] + 8), fill=(55, 45, 35, 255), width=5)
        draw.line((cat_center[0] - 72, cat_center[1] + 18, cat_center[0] - 120, cat_center[1] + 32), fill=(55, 45, 35, 255), width=5)
    for dx in (72, 48):
        draw.line((cat_center[0] + 72, cat_center[1] + 18, cat_center[0] + 120, cat_center[1] + 8), fill=(55, 45, 35, 255), width=5)
        draw.line((cat_center[0] + 72, cat_center[1] + 18, cat_center[0] + 120, cat_center[1] + 32), fill=(55, 45, 35, 255), width=5)

    # text: 毛GO
    text = '毛GO'
    font = load_font(110)
    # Shadow / outline effect to resemble the provided mark
    for offset in [(-3, -3), (3, -3), (-3, 3), (3, 3)]:
        draw.text((95 + offset[0], 365 + offset[1]), text, font=font, fill=(255, 235, 182, 255), stroke_width=4, stroke_fill=(255, 177, 30, 255))
    draw.text((95, 365), text, font=font, fill=(255, 255, 255, 255), stroke_width=0)

    img.save(path)


if __name__ == '__main__':
    out = str(Path(__file__).resolve().parent / 'pet.png')
    create_pet_icon(out)
    print('created', out)
