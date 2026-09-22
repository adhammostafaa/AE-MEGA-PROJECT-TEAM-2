import os
import csv
import math
import random
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# shapes that are required in the project
shapes = ["circle", "semicircle", "quarter_circle", "triangle",
          "rectangle", "pentagon", "star", "cross"]

# colors required in the project
colors = {
    "white":  (255, 255, 255),
    "black":  (20, 20, 20),
    "red":    (200, 30, 30),
    "blue":   (30, 80, 200),
    "green":  (30, 150, 60),
    "purple": (140, 60, 190),
    "brown":  (120, 75, 35),
    "orange": (230, 120, 20),
}

# all alphabets and numbers
chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
img_size = 640  # i chose this image size specifically because most odlc dataset images had this img size
shape_canvas = 140  # the size of the shape and number/letter inside before shrinking
obj_min = 30  # the range of the sizes of the shapes after shrinking for randomness
obj_max = 70 
                       
font_candidates = ["C:/Windows/Fonts/arialbd.ttf",
                   "/Library/Fonts/Arial Bold.ttf"]

def load_font(size):
    for path in font_candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

# this function makes the background for the image
def make_background(size):
    base = random.choice([(70, 120, 45), (110, 90, 65), (130, 130, 130)]) # color for the base (grass or dirt or pavement)
    bg = Image.new("RGB", size, base) # creates a new rgb image
    draw = ImageDraw.Draw(bg) 
    for _ in range(60):
        shade = tuple(
            max(0, min(255, channel + random.randint(-15, 15)))
            for channel in base) # makes the base slightly darker or lighter
        r = random.randint(20, 50)                                      #}
        x, y = random.randint(0, size[0]), random.randint(0, size[1])   #} <- draw random sized and shaded circles 
        draw.ellipse((x - r, y - r, x + r, y + r), fill=shade)          #}

    bg = bg.filter(ImageFilter.GaussianBlur(4)) # blur

    arr = np.array(bg).astype(np.int16)
    arr += np.random.normal(0, 8, arr.shape).astype(np.int16)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)) # add more randomness by changing some pixels

# this function should draw the shapes
def draw_shape(draw, shape, color, size):
    c = size / 2 # the centre of the drawing area
    if shape == "circle":
        draw.ellipse((0, 0, size, size), fill=color) # draws a circle
    elif shape == "semicircle":
        draw.pieslice((0, 0, size, size), 0, 180, fill=color) # draws a semicircle
    elif shape == "quarter_circle":
        draw.pieslice((0, 0, size, size), 270, 360, fill=color) # draws a quarter circle
    elif shape == "triangle":
        draw.polygon([(c, 0), (0, size), (size, size)], fill=color) # draws a triangle
    elif shape == "rectangle":
        draw.rectangle((0, 0, size, size), fill=color) # draws a rectangle
    elif shape == "pentagon":
        pts = [(c + c * math.cos(math.radians(90 + i * 72)),
                c - c * math.sin(math.radians(90 + i * 72)))
                for i in range(5)]
        draw.polygon(pts, fill=color) # draws pentagon
    elif shape == "star":
        pts = []
        for i in range(10):
            r = c if i % 2 == 0 else c * 0.45
            ang = math.radians(90 + i * 36)
            pts.append((c + r * math.cos(ang), c - r * math.sin(ang)))
        draw.polygon(pts, fill=color) # draws star
    elif shape == "cross":
        t = size * 0.28
        draw.rectangle((c - t / 2, 0, c + t / 2, size), fill=color)
        draw.rectangle((0, c - t / 2, size, c + t / 2), fill=color) # draws cross

# this function creates the complete object (the shape with letter or number inside)
def generate_object():
    shape = random.choice(shapes) # randomly chooses a shape
    shape_color_name = random.choice(list(colors)) # randomly chooses color for the shape
    char_color_name = random.choice([c for c in colors if c != shape_color_name]) # randomly chooses color for the number
    character = random.choice(chars) # randomly chooses letter or number

    canvas = Image.new("RGBA", (shape_canvas, shape_canvas), (0, 0, 0, 0)) # creates empty image where the object is created
    draw = ImageDraw.Draw(canvas)

    draw_shape(draw, shape, colors[shape_color_name] + (255,), shape_canvas) # draws the shape

    font = load_font(int(shape_canvas * 0.5))
    bbox = draw.textbbox((0, 0), character, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1] # calculate the char's width and height and centre
    center = shape_canvas / 2
    draw.text((center - text_w / 2 - bbox[0], center - text_h / 2 - bbox[1]),
               character, font=font, fill=colors[char_color_name] + (255,)) # this draws the character in the center of the shape

    angle = random.uniform(0, 360)
    canvas = canvas.rotate(angle, resample=Image.BICUBIC, expand=True) # randomly rotates the object

    target = random.randint(obj_min, obj_max)
    scale = target / canvas.width
    new_size = (max(1, int(canvas.width * scale)), max(1, int(canvas.height * scale)))
    canvas = canvas.resize(new_size, Image.LANCZOS) # the object is shrunk to size between 30 and 70 pixels

    meta = {
        "shape": shape,
        "shape_color": shape_color_name,
        "character": character,
        "character_color": char_color_name,
    } # this returns the shape image and the details (shape, color, char and char color)
    return canvas, meta

# this function creates new background and new object (random)
def compose_image():
    bg = make_background((img_size, img_size))
    obj, meta = generate_object()

    max_x = img_size - obj.width          #}  
    max_y = img_size - obj.height         #} <- creates random position for the object in the final image
    x = random.randint(0, max(1, max_x))  #}
    y = random.randint(0, max(1, max_y))  #}

    bg.paste(obj, (x, y), obj) # puts the object in the image

    alpha = np.array(obj.split()[-1])
    ys, xs = np.where(alpha > 10)
    x0, y0 = x + int(xs.min()), y + int(ys.min())
    x1, y1 = x + int(xs.max()), y + int(ys.max())

    img = bg
    if random.random() < 0.4: # adds blur to some images for better generalization
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))

    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.75, 1.25)) # random brightness
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.15)) # random contrast

    arr = np.array(img).astype(np.int16)                         #}
    arr += np.random.normal(0, 6, arr.shape).astype(np.int16)    #} <- adds noise to the image
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)) #}
 
    yolo_box = (
        (x0 + x1) / 2 / img_size,
        (y0 + y1) / 2 / img_size,
        (x1 - x0) / img_size,
        (y1 - y0) / img_size,
    )
    return img, yolo_box, meta

# this function creates the dataset
# root is the output folder
# n_total is the number of images
# split is the percentage of train, validate and test
def build_dataset(root, n_total, split=(0.8, 0.1, 0.1), seed=None):
    if seed is not None: # if seed is given, then the same random images will be created everytime
        random.seed(seed)
        np.random.seed(seed)
    n_train = int(n_total * split[0]) # calculates num of images going to train folder
    n_val = int(n_total * split[1]) # "" "" "" "" "" "" validate folder
    n_test = n_total - n_train - n_val # "" "" "" "" "" "" test folder
    counts = {"train": n_train, "val": n_val, "test": n_test}

    # creates the folders
    for name in counts:
        os.makedirs(f"{root}/images/{name}", exist_ok=True)
        os.makedirs(f"{root}/labels/{name}", exist_ok=True)

    meta_rows = []
    done = 0
    # looping through each dataset split
    for split_name, count in counts.items():
        for i in range(count):
            # generate image
            img, yolo_box, meta = compose_image()
            fname = f"{split_name}_{i:05d}" #create filename
            img.save(f"{root}/images/{split_name}/{fname}.png") # save image

            class_id = shapes.index(meta["shape"])
            with open(f"{root}/labels/{split_name}/{fname}.txt", "w") as f:
                f.write(f"{class_id} " + " ".join(f"{v:.6f}" for v in yolo_box) + "\n")

            meta.update({"split": split_name, "file": f"{fname}.png"})
            meta_rows.append(meta)
            done += 1
            if done % 100 == 0:
                print(f"generated {done}/{n_total}")

    with open(f"{root}/metadata.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file", "split", "shape", "shape_color", "character", "character_color"])
        writer.writeheader()
        writer.writerows(meta_rows)
    with open(f"{root}/classes.txt", "w") as f:
        f.write("\n".join(shapes) + "\n")
    print(f"{n_total} images written to {root}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1200, help="total images to generate")
    parser.add_argument("--out", type=str, default="dataset", help="output folder")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_dataset(args.out, args.n, seed=args.seed)


# to run the program locally on your computer
# run cmd in the folder where the code is downloaded
# and paste this command: python your_script.py --n 2000 --out odlc_dataset --seed 42
# dataset folder will be created inside the same folder