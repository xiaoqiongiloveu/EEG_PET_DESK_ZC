import os
import subprocess
import sys

cwd = os.getcwd()
venv_py = os.path.join(cwd, '.venv', 'Scripts', 'python.exe')
if not os.path.exists(venv_py):
    print('Virtualenv python not found:', venv_py)
    sys.exit(2)

spec_path = os.path.join(cwd, 'python_ads1299', 'main_uart_v1.2.spec')
if not os.path.exists(spec_path):
    print('Spec file not found:', spec_path)
    sys.exit(3)

# Read current version (or default to 1.2)
vf = os.path.join(cwd, 'python_ads1299', 'VERSION')
if os.path.exists(vf):
    try:
        with open(vf, 'r') as f:
            cur_ver = float(f.read().strip())
    except Exception as e:
        print('Failed to parse VERSION:', e)
        sys.exit(4)
else:
    cur_ver = 1.2

print('Current VERSION:', cur_ver)

# Run PyInstaller
cmd = [venv_py, '-m', 'PyInstaller', '--noconfirm', '--distpath', '.', '--workpath', os.path.join('.', 'build'), spec_path]
print('Running:', ' '.join(cmd))
res = subprocess.run(cmd)
if res.returncode != 0:
    print('PyInstaller failed with code', res.returncode)
    sys.exit(res.returncode)

# Produced exe from spec name
produced = os.path.join(cwd, 'main_uart_v1.2.exe')
if not os.path.exists(produced):
    print('Produced exe not found:', produced)
    sys.exit(5)

# Rename produced to include current VERSION
target_name = f'main_uart_v{cur_ver}.exe'
target = os.path.join(cwd, target_name)
try:
    if os.path.exists(target):
        print('Target already exists, will overwrite:', target)
        os.remove(target)
    os.replace(produced, target)
    print('Renamed produced exe to:', target_name)
except Exception as e:
    print('Failed to rename produced exe:', e)
    sys.exit(6)

# Bump version by 0.1 and write back
new_ver = round(cur_ver + 0.1, 1)
try:
    with open(vf, 'w') as f:
        f.write(str(new_ver))
    print(f'Version bumped: {cur_ver} -> {new_ver}')
except Exception as e:
    print('Failed to write VERSION:', e)
    sys.exit(7)

print('Build and bump completed.')
