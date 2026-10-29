from PIL import Image, ImageDraw
import os
import sys


def fallback_icon(path):
    im = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.polygon([(128, 20), (220, 128), (128, 236), (36, 128)], fill=(200, 200, 200, 255), outline=(0, 0, 0))
    d.ellipse((110, 90, 146, 126), fill=(255, 255, 255, 255), outline=(0, 0, 0))
    sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
    im.save(path, sizes=sizes)
    return path


def find_source_image(folder, explicit_path=None):
    if explicit_path and os.path.exists(explicit_path):
        return os.path.abspath(explicit_path)

    candidates = [
        'pet.png', 'pet.jpg', 'pet.jpeg', 'cute_pet.png', 'cute_pet.jpg',
        'mouse.png', 'mouse.jpg', 'mouse.jpeg', 'icon.png', 'icon.jpg', 'icon.jpeg',
        'logo.png', 'logo.jpg', 'logo.jpeg'
    ]
    for name in candidates:
        full = os.path.join(folder, name)
        if os.path.exists(full):
            return full

    for entry in sorted(os.listdir(folder)):
        lower = entry.lower()
        if lower.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
            return os.path.join(folder, entry)
    return None


def main():
    folder = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(folder, 'mouse.ico')
    explicit_path = sys.argv[1] if len(sys.argv) > 1 else None
    src = find_source_image(folder, explicit_path)

    if src:
        img = Image.open(src).convert('RGBA')
        resample = getattr(Image, 'Resampling', Image).LANCZOS
        img = img.resize((256, 256), resample)
        img.save(out_path, format='ICO')
        print('mouse.ico created from image:', os.path.abspath(src))
    else:
        fallback_icon(out_path)
        print('No source image found; fallback mouse.ico generated at', os.path.abspath(out_path))

    print('mouse.ico created at', os.path.abspath(out_path))


if __name__ == '__main__':
    main()
