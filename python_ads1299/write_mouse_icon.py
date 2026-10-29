# Generates a simple 16x16 32-bit ICO file (solid blue) without dependencies
# Writes mouse.ico in the current directory

def make_ico(path='mouse.ico'):
    import struct

    width = 16
    height = 16
    # ICONDIR: reserved, type (1=icon), count
    icondir = struct.pack('<HHH', 0, 1, 1)

    # ICONDIRENTRY: width, height, colorcount, reserved, planes, bitcount, bytesinres, imageoffset
    # We'll compute bytesinres and offset after building image data

    # Build BITMAPINFOHEADER
    biSize = 40
    biWidth = width
    biHeight = height * 2  # bitmap + mask
    biPlanes = 1
    biBitCount = 32
    biCompression = 0
    biSizeImage = 0
    biXPelsPerMeter = 0
    biYPelsPerMeter = 0
    biClrUsed = 0
    biClrImportant = 0
    bmp_header = struct.pack('<IIIHHIIIIII', biSize, biWidth, biHeight, biPlanes, biBitCount, biCompression, biSizeImage, biXPelsPerMeter, biYPelsPerMeter, biClrUsed, biClrImportant)

    # Pixel data: bottom-up rows, BGRA for each pixel
    # Create a simple solid-blue RGBA (B=255,G=0,R=0,A=255)
    pixel = bytes([255, 0, 0, 255])
    row = pixel * width  # 4*width bytes per row
    pixels = b''.join([row for _ in range(height)])  # height rows

    # AND mask: 1 bit per pixel, padded to 32-bit per scanline
    # For width=16, bits per row=16 -> bytes=2, padded to 4
    mask_row = b'\x00\x00\x00\x00'  # 4 bytes per row (all zeros)
    mask = mask_row * height

    image_data = bmp_header + pixels + mask
    bytes_in_res = len(image_data)

    # ICONDIRENTRY fields
    width_byte = width if width < 256 else 0
    height_byte = height if height < 256 else 0
    colorcount = 0
    reserved = 0
    planes = 1
    bitcount = 32

    # image offset (after icondir + one entry): 6 + 16 = 22
    image_offset = 6 + 16

    icondirentry = struct.pack('<BBBBHHII', width_byte, height_byte, colorcount, reserved, planes, bitcount, bytes_in_res, image_offset)

    with open(path, 'wb') as f:
        f.write(icondir)
        f.write(icondirentry)
        f.write(image_data)

    print('Wrote', path, 'size', bytes_in_res + image_offset)

if __name__ == '__main__':
    make_ico()
