import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# =========================
# SETTINGS
# =========================

WIDTH = 900
HEIGHT = 900
MAX_ITER = 300

# Mode: "julia" or "mandelbrot"
MODE = "julia"

# Julia Constant
C_REAL = -0.7
C_IMAG = 0.27015

# View Window
xmin, xmax = -2, 2
ymin, ymax = -2, 2

# =========================
# FAST FRACTAL ENGINE
# =========================

@njit(fastmath=True)
def compute_fractal(xmin, xmax, ymin, ymax, width, height, max_iter, mode, c_real, c_imag):

    image = np.zeros((height, width))

    for y in range(height):
        zy = ymin + (y / height) * (ymax - ymin)

        for x in range(width):
            zx = xmin + (x / width) * (xmax - xmin)

            if mode == 0:
                c = complex(zx, zy)
                z = 0
            else:
                c = complex(c_real, c_imag)
                z = complex(zx, zy)

            iteration = 0

            while abs(z) <= 2 and iteration < max_iter:
                z = z*z + c
                iteration += 1

            # Smooth coloring
            if iteration < max_iter:
                log_zn = np.log(z.real*z.real + z.imag*z.imag) / 2
                nu = np.log(log_zn / np.log(2)) / np.log(2)
                iteration = iteration + 1 - nu

            image[y, x] = iteration

    return image


# =========================
# DRAW FUNCTION
# =========================

def draw():

    global xmin, xmax, ymin, ymax

    mode_flag = 0 if MODE == "mandelbrot" else 1

    fractal = compute_fractal(
        xmin, xmax, ymin, ymax,
        WIDTH, HEIGHT,
        MAX_ITER,
        mode_flag,
        C_REAL, C_IMAG
    )

    plt.clf()
    plt.imshow(fractal, cmap="plasma", extent=[xmin, xmax, ymin, ymax])
    plt.title(f"{MODE.upper()} SET — Scroll = Zoom | Right Click = Save")
    plt.axis("off")
    plt.draw()


# =========================
# MOUSE INTERACTION
# =========================

def zoom(event):

    global xmin, xmax, ymin, ymax

    scale = 0.8 if event.button == 'up' else 1.25

    mx, my = event.xdata, event.ydata

    if mx is None or my is None:
        return

    width = xmax - xmin
    height = ymax - ymin

    xmin = mx - width * scale / 2
    xmax = mx + width * scale / 2
    ymin = my - height * scale / 2
    ymax = my + height * scale / 2

    draw()


def save_image(event):

    if event.button == 3:
        plt.savefig("fractal_HD.png", dpi=300, bbox_inches="tight")
        print("✅ Saved as fractal_HD.png")


# =========================
# ANIMATED ZOOM
# =========================

def auto_zoom(frames=60):

    global xmin, xmax, ymin, ymax

    for i in range(frames):

        factor = 0.92

        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2

        width = (xmax - xmin) * factor
        height = (ymax - ymin) * factor

        xmin = cx - width/2
        xmax = cx + width/2
        ymin = cy - height/2
        ymax = cy + height/2

        draw()
        plt.pause(0.05)


# =========================
# MAIN
# =========================

plt.figure(figsize=(8, 8))

draw()

plt.connect('scroll_event', zoom)
plt.connect('button_press_event', save_image)

print("Controls:")
print("🖱 Scroll Wheel = Zoom")
print("🖱 Right Click = Save HD Image")
print("⌨ Close Window To Exit")

# Uncomment for automatic zoom animation
# auto_zoom(80)

plt.show()
