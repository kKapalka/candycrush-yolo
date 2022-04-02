
import numpy as np
import cv2
import random
import os

def noisy_pixel():
    return [random.random()*256,random.random()*256,random.random()*256,255]

pixel = [random.random()*256,random.random()*256,random.random()*256,255]

image_number = 1

background_count_per_type = 2
foreground_count = 10

def shift_pixel():
    global pixel
    pixel = [(pixel[0] + random.randrange(0, 5)) % 256, (pixel[1] + random.randrange(0, 4)) % 256, (pixel[2] + random.randrange(0, 3)) % 256, 255]
    return pixel

def create_yolo_entry(image_width, image_height, box_xmin, box_ymin, box_width, box_height):
    xpos_norm, ypos_norm = (box_xmin + (box_width / 2)) / image_width, (box_ymin + (box_height/2)) / image_height
    box_w_norm, box_h_norm = box_width / image_width, box_height / image_height
    return str(index)+' '+str(xpos_norm)+' '+str(ypos_norm)+' '+str(box_w_norm)+' '+str(box_h_norm)
    

def merge_image(back, front, x,y):
    # convert to rgba
    if back.shape[2] == 3:
        back = cv2.cvtColor(back, cv2.COLOR_BGR2BGRA)
    if front.shape[2] == 3:
        front = cv2.cvtColor(front, cv2.COLOR_BGR2BGRA)

    # crop the overlay from both images
    bh,bw = back.shape[:2]
    fh,fw = front.shape[:2]
    x1, x2 = max(x, 0), min(x+fw, bw)
    y1, y2 = max(y, 0), min(y+fh, bh)
    front_cropped = front[y1-y:y2-y, x1-x:x2-x]
    back_cropped = back[y1:y2, x1:x2]

    alpha_front = front_cropped[:,:,3:4] / 255
    alpha_back = back_cropped[:,:,3:4] / 255
    
    # replace an area in result with overlay
    result = back.copy()
    result[y1:y2, x1:x2, :3] = alpha_front * front_cropped[:,:,:3] + (1-alpha_front) * back_cropped[:,:,:3]
    result[y1:y2, x1:x2, 3:4] = (alpha_front + alpha_back) / (1 + alpha_front*alpha_back) * 255

    return result

def load_candies():
    candy_set = []
    path = 'candies/'
    files = os.listdir(path)
    files.sort()
    currentFilename = ''
    for file in files:
        image = cv2.imread(path+file, cv2.IMREAD_UNCHANGED)
        if(image is not None):
            imageData = image.copy().astype(float)
            if(currentFilename != file.split('(')[0]):
                candy_set.append([imageData])
                currentFilename = file.split('(')[0]
            else:
                candy_set[-1].append(imageData)
    return candy_set

candies = load_candies()
classes = []
width, height = 360, 360

noise_background_set = []
blurred_noisy_background_set = []
uniform_background_set = []
                                
candy_foreground_set = []
candy_classification_set = []

empty_template = np.array([[[0,0,0,0] for x in range(0, width)] for y in range(0, height)]).astype(float)

for i in range(0, background_count_per_type):
    noise_background_set.append(np.array([[noisy_pixel() for x in range(0, width)] for y in range(0, height)]).astype(float))
    pixel = noisy_pixel()
    blurred_noisy_background_set.append(np.array([[shift_pixel() for x in range(0, width)] for y in range(0, height)]).astype(float))
    b = random.randrange(0, 256)
    g = random.randrange(0, 256)
    r = random.randrange(0, 256)
    uniform_background_set.append(np.array([[[b,g,r,255] for x in range(0, width)] for y in range(0, height)]).astype(float))


def generate_image(background):
    global image_number
    for foreground in candy_foreground_set:
        final_filename = "img"+str(image_number)
        rand = random.random()
        destination = "training_imgs/train"
        if(rand > 0.85):
            destination = "training_imgs/val"
        with open(destination+"/labels/"+final_filename+".txt", 'w') as f:
            f.write('\n'.join(foreground[1]))
        image = merge_image(background, foreground[0], 0, 0)
        if not cv2.imwrite(destination+"/images/"+final_filename+".png", image):
            raise Exception("could not write")
        image_number=image_number+1
    for i in range(0, int(len(candy_foreground_set)/20)):
        final_filename = "img"+str(image_number)
        if not cv2.imwrite(destination+"/images/"+final_filename+".png", background):
            raise Exception("could not write")
        with open(destination+"/labels/"+final_filename+".txt", 'w') as f:
                f.write('\n')
        image_number=image_number+1
                                
for i in range(0, foreground_count):
    lines = []
    template = empty_template.copy()
    populated_areas = np.zeros([width, height])
    max_candy_amount = random.randrange(7, 13)
    for j in range(0, max_candy_amount):
        index = random.randrange(0, len(candies))
        image_to_add = candies[index][random.randrange(0, len(candies[index]))]
        h, w, _ = image_to_add.shape
        x, y = width-w, height-h
        xpos, ypos = random.randrange(0, x), random.randrange(0, y)
        while 1 in populated_areas[xpos+int(w/3):xpos+w-int(w/3),ypos+int(h/3):ypos+h-int(h/3)]:
            xpos, ypos = random.randrange(0, x), random.randrange(0, y)
        lines.append(create_yolo_entry(width, height, xpos, ypos, w, h))
        populated_areas[xpos:xpos+w,ypos:ypos+h] = 1
        template = merge_image(template, image_to_add, xpos, ypos)
    candy_foreground_set.append((template, lines))

for background in noise_background_set:
    generate_image(background)
    
for background in uniform_background_set:
    generate_image(background)

for background in blurred_noisy_background_set:
    generate_image(background)




