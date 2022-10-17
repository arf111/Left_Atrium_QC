import numpy as np


class CropPad(object):
    def __init__(self, h, w, chw=False):
        '''
        if image > taget image size, simply cropped
        otherwise, pad image to target size.
        :param h: target image height
        :param w: target image width
        '''
        self.target_h = h
        self.target_w = w
        self.chw = chw

    def __call__(self, img):
        # center padding/cropping
        if len(img.shape) == 3:
            if self.chw:
                x, y = img.shape[1], img.shape[2]
            else:
                x, y = img.shape[0], img.shape[1]
        else:
            x, y = img.shape[0], img.shape[1]

        x_s = (x - self.target_h) // 2
        y_s = (y - self.target_w) // 2
        x_c = (self.target_h - x) // 2
        y_c = (self.target_w - y) // 2
        if len(img.shape) == 2:

            if x > self.target_h and y > self.target_w:
                slice_cropped = img[x_s:x_s + self.target_h, y_s:y_s + self.target_w]
            else:
                slice_cropped = np.zeros((self.target_h, self.target_w), dtype=img.dtype)
                if x <= self.target_h and y > self.target_w:
                    slice_cropped[x_c:x_c + x, :] = img[:, y_s:y_s + self.target_w]
                elif x > self.target_h > 0 and y <= self.target_w:
                    slice_cropped[:, y_c:y_c + y] = img[x_s:x_s + self.target_h, :]
                else:
                    slice_cropped[x_c:x_c + x, y_c:y_c + y] = img[:, :]
        if len(img.shape) == 3:
            if not self.chw:
                if x > self.target_h and y > self.target_w:
                    slice_cropped = img[x_s:x_s + self.target_h, y_s:y_s + self.target_w, :]
                else:
                    slice_cropped = np.zeros((self.target_h, self.target_w, img.shape[2]), dtype=img.dtype)
                    if x <= self.target_h and y > self.target_w:
                        slice_cropped[x_c:x_c + x, :, :] = img[:, y_s:y_s + self.target_w, :]
                    elif x > self.target_h > 0 and y <= self.target_w:
                        slice_cropped[:, y_c:y_c + y, :] = img[x_s:x_s + self.target_h, :, :]
                    else:
                        slice_cropped[x_c:x_c + x, y_c:y_c + y, :] = img
            else:
                if x > self.target_h and y > self.target_w:
                    slice_cropped = img[:, x_s:x_s + self.target_h, y_s:y_s + self.target_w]
                else:
                    slice_cropped = np.zeros((img.shape[0], self.target_h, self.target_w), dtype=img.dtype)
                    if x <= self.target_h and y > self.target_w:
                        slice_cropped[:, x_c:x_c + x, :] = img[:, :, y_s:y_s + self.target_w]
                    elif x > self.target_h > 0 and y <= self.target_w:
                        slice_cropped[:, :, y_c:y_c + y] = img[:, x_s:x_s + self.target_h, :]
                    else:
                        slice_cropped[:, x_c:x_c + x, y_c:y_c + y] = img

        return slice_cropped

    def __repr__(self):
        return self.__class__.__name__ + 'padding to ({0}, {1})'. \
            format(self.target_h, self.target_w)
