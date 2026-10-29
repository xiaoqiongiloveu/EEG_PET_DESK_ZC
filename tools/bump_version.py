import os
import sys

vf = os.path.join(os.getcwd(), "python_ads1299", "VERSION")
if not os.path.exists(vf):
    print("VERSION missing")
    sys.exit(1)

with open(vf, "r") as f:
    try:
        cur = float(f.read().strip())
    except Exception as e:
        print("Failed to parse VERSION:", e)
        sys.exit(2)

new = round(cur + 0.1, 1)
with open(vf, "w") as f:
    f.write(str(new))

old = os.path.join(os.getcwd(), f"main_uart_v{cur}.exe")
newexe = os.path.join(os.getcwd(), f"main_uart_v{new}.exe")
if os.path.exists(old):
    try:
        os.replace(old, newexe)
        print(f"Moved: {old} -> {newexe}")
    except Exception as e:
        print("Rename failed:", e)
        sys.exit(3)
else:
    print("Old exe not found:", old)

print(f"VERSION updated: {cur} -> {new}")
