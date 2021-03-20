import PySimpleGUI as sg
import PIL
from PIL import Image
import io
import base64
import os
import numpy as np

"""
    Using PIL with PySimpleGUI
    This image viewer uses both a thumbnail creation function and an image resizing function that
    you may find handy to include in your code.
    Copyright 2020 PySimpleGUI.org
"""

THUMBNAIL_SIZE = (200,200)
IMAGE_SIZE = (800,800)
THUMBNAIL_PAD = (1,1)
ROOT_FOLDER_IMAGES = r'../data/VOC2012/JPEGImages'
ROOT_FOLDER_TEXTFILES = r'../data/scores/' #r'../data/scores/concat_pred.npy'




screen_size = sg.Window.get_screen_size()
thumbs_per_row = int(screen_size[0]/(THUMBNAIL_SIZE[0]+THUMBNAIL_PAD[0])) - 1
thumbs_rows = int(screen_size[1]/(THUMBNAIL_SIZE[1]+THUMBNAIL_PAD[1])) - 1
THUMBNAILS_PER_PAGE = (thumbs_per_row, thumbs_rows)

class_preds = np.load('../data/scores/concat_pred.npy')
filenames = np.load('../data/scores/filenames.npy', allow_pickle = True)
filenames = list(np.concatenate(filenames).flat)

classes = ['aeroplane', 'bicycle', 'bird', 'boat',
'bottle', 'bus', 'car', 'cat', 'chair',
'cow', 'diningtable', 'dog', 'horse',
'motorbike', 'person', 'pottedplant',
'sheep', 'sofa', 'train',
'tvmonitor']



def make_square(im, min_size=256, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im


def convert_to_bytes(file_or_bytes, resize=None, fill=False):
    '''
    Will convert into bytes and optionally resize an image that is a file or a base64 bytes object.
    Turns into  PNG format in the process so that can be displayed by tkinter
    :param file_or_bytes: either a string filename or a bytes base64 image object
    :type file_or_bytes:  (Union[str, bytes])
    :param resize:  optional new size
    :type resize: (Tuple[int, int] or None)
    :return: (bytes) a byte-string object
    :rtype: (bytes)
    '''
    if isinstance(file_or_bytes, str):
        img = PIL.Image.open(file_or_bytes)
    else:
        try:
            img = PIL.Image.open(io.BytesIO(base64.b64decode(file_or_bytes)))
        except Exception as e:
            dataBytesIO = io.BytesIO(file_or_bytes)
            img = PIL.Image.open(dataBytesIO)

    cur_width, cur_height = img.size
    if resize:
        new_width, new_height = resize
        scale = min(new_height / cur_height, new_width / cur_width)
        img = img.resize((int(cur_width * scale), int(cur_height * scale)), PIL.Image.ANTIALIAS)
    if fill:
        img = make_square(img, THUMBNAIL_SIZE[0])
    with io.BytesIO() as bio:
        img.save(bio, format="PNG")
        del img
        return bio.getvalue()




def display_image_window(filename):
    try:
        layout = [[sg.Image(data=convert_to_bytes(filename, IMAGE_SIZE), enable_events=True)]]
        e,v = sg.Window(filename, layout, modal=True, element_padding=(0,0), margins=(0,0)).read(close=True)
    except Exception as e:
        print(f'** Display image error **', e)
        return


def make_thumbnails(flist):
    layout = [[]]
    for row in range(THUMBNAILS_PER_PAGE[1]):
        row_layout = []
        for col in range(THUMBNAILS_PER_PAGE[0]):
            try:
                f = flist[row*THUMBNAILS_PER_PAGE[1] + col]
                # row_layout.append(sg.B(image_data=convert_to_bytes(f, THUMBNAIL_SIZE), k=(row,col), pad=THUMBNAIL_PAD))
                row_layout.append(sg.B('',k=(row,col), size=(0,0), pad=THUMBNAIL_PAD,))
            except:
                pass
        layout += [row_layout]
    layout += [[sg.B(sg.SYMBOL_LEFT + ' Prev', size=(10,3), k='-PREV-'),
              sg.B('Next '+sg.SYMBOL_RIGHT, size=(10,3), k='-NEXT-'),
              sg.B('Exit', size=(10,3)), sg.B('aeroplane'), sg.B('bicycle'),
              sg.B('bird'), sg.B('boat'), sg.B('bottle'), sg.B('bus'), sg.B('car'),
              sg.B('cat'), sg.B('chair'), sg.B('cow'), sg.B('diningtable'),
              sg.B('dog'), sg.B('horse'), sg.B('motorbike'), sg.B('person'),
              sg.B('pottedplant'), sg.B('sheep'), sg.B('sofa'), sg.B('train'),
              sg.B('tvmonitor'),]]

    return sg.Window('Thumbnails', layout, element_padding=(0, 0), margins=(0, 0),\
           finalize=True, grab_anywhere=False, location=(0,0), return_keyboard_events=True)

EXTS = ('png', 'jpg', 'gif')

def display_images(t_win, offset, files):
    currently_displaying = {}
    row = col = 0
    while True:
        if offset + 1 > len(files) or row == THUMBNAILS_PER_PAGE[1]:
            break
        f = files[offset]
        currently_displaying[(row, col)] = f
        try:
            t_win[(row, col)].update(image_data=convert_to_bytes(f, THUMBNAIL_SIZE, True))
        except Exception as e:
            print(f'Error on file: {f}', e)
        col = (col + 1) % THUMBNAILS_PER_PAGE[0]
        if col == 0:
            row += 1

        offset += 1
    if not (row == 0 and col == 0):
        while row != THUMBNAILS_PER_PAGE[1]:
            t_win[(row, col)].update(image_data=sg.DEFAULT_BASE64_ICON)
            currently_displaying[(row, col)] = None
            col = (col + 1) % THUMBNAILS_PER_PAGE[0]
            if col == 0:
                row += 1


    return offset, currently_displaying


def main():
    class_preds_i = class_preds[:,0]
    class_preds_i, class_files = zip(*sorted(zip(class_preds_i, filenames)))
    class_files = np.flip(class_files)
    #print(class_files)
    #print(class_preds_i)

    files = [os.path.join(ROOT_FOLDER_IMAGES, f + '.jpg') for f in class_files]
    files.sort()

    t_win = make_thumbnails(files)
    offset, currently_displaying = display_images(t_win, 0, files)

    # offset = THUMBNAILS_PER_PAGE[0] * THUMBNAILS_PER_PAGE[1]
    # currently_displaying = {}
    while True:
        win, event, values = sg.read_all_windows()
        print(event, values)
        if win == sg.WIN_CLOSED:            # if all windows are closed
            break

        if event == sg.WIN_CLOSED or event == 'Exit':
            break

        if isinstance(event, tuple):
            display_image_window(currently_displaying.get(event))
            continue

        if event in classes:
            print(event)
            indx = classes.index(event)
            print(indx)
            class_preds_i = class_preds[:, indx]
            print(class_preds)
            class_preds_i, class_files = zip(*sorted(zip(class_preds_i, filenames)))
            class_files = np.flip(class_files)

            files = [os.path.join(ROOT_FOLDER_IMAGES, f + '.jpg') for f in class_files]
            files.sort()

            offset = 0
            offset, currently_displaying = display_images(t_win, offset, files)

        if event == '-NEXT-' or event.endswith('Down'):
            offset, currently_displaying = display_images(t_win, offset, files)
        elif event == '-PREV-' or event.endswith('Up'):
            offset -= THUMBNAILS_PER_PAGE[0]*THUMBNAILS_PER_PAGE[1]*2
            if offset < 0:
                offset = 0
            offset, currently_displaying = display_images(t_win, offset, files)





if __name__ == '__main__':
    main()
