from scipy import ndimage
import cv2
import numpy as np

## PyTorch
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
from scipy.linalg import expm, sinm, cosm, logm


class SymNIST(data.Dataset):
  def __init__(self, size, source, rotation_type, rotation_std, blurred=0, frame=40, rotation=0, translation=0, scale=1, seed=42):

    """
    Inputs:
      size - number of data points
      frame - size of square augmented data
      source - (x_train, y_train)
      rotation - max/characteristic rotation angle
      translation - max/characteristic translation distance
      scale - max/characteristic scaling factor
    """

    super().__init__()
    self.size = size
    self.frame = frame
    self.r = rotation
    self.rotation_type = rotation_type
    self.rotation_std = rotation_std
    self.t = translation
    self.s = scale
    self.source_data, self.source_label = source[0][:self.size], source[1][:self.size]
    self.blurred = blurred

    torch.manual_seed(seed)
    np.random.seed(seed)
    self.generate()

  def generate(self):
    """
    Each data point is an MNIST image placed in a larger canvas of size 
    frame x frame.
    """
    n = (self.frame - self.source_data.shape[1])/2
    n = int(n)
    padded_data = np.pad(self.source_data, ((0,0),(n,n),(n,n)), 'constant')/256
    padded_label = np.pad(self.source_label, ((0,0),(n,n),(n,n)), 'constant')/256
    r_list = np.zeros(self.size)
    t_list = np.zeros((self.size,2))
    s_list = np.zeros(self.size)


    if self.blurred>0:
      for img in range(self.size):
        padded_data[img] = cv2.GaussianBlur(padded_data[img], (self.blurred,self.blurred), 0)
        padded_label[img] = cv2.GaussianBlur(padded_label[img], (self.blurred,self.blurred), 0)


    # Scaling Generator
    if self.s != 1:
      # Prepare list of scalings
      if self.rotation_type == "normal":
        s_list = np.random.normal(self.s, self.rotation_std, self.size)
      if self.rotation_type == "uniform":
        if self.rotation_std == 0:
          s_list = np.random.uniform(1/self.s, self.s, self.size) 
        elif self.rotation_std == 1:
          s_list = np.random.uniform(1, self.s, self.size) 
        elif self.rotation_std == -1:
          s_list = np.random.uniform(1/self.s, 1, self.size) 
      for img in range(self.size):
        s_i = s_list[img]
        padded_data[img] = cv2_clipped_zoom(padded_data[img], s_i)

    # Rotation Generator
    if self.r > 0:
      # Prepare list of rotations
      if self.rotation_type == "normal":
         r_list = np.random.normal(self.r, self.rotation_std, self.size)
      if self.rotation_type == "uniform":
         if self.rotation_std == 0:
            r_list = np.random.uniform(-1/2*np.abs(self.r), 1/2*np.abs(self.r), self.size)
         elif self.rotation_std == 1:
           if self.r>=0:
              r_list = np.random.uniform(0, self.r, self.size)
           elif self.r<0:
              r_list = np.random.uniform(self.r, 0, self.size)
         elif self.rotation_std == -1:
            r_list = np.random.uniform(-1/4*np.abs(self.r), 3/4*np.abs(self.r), self.size)
      for img in range(self.size):
        r_i = r_list[img]
        padded_data[img] = ndimage.rotate(padded_data[img], r_i, 
                                          reshape=False)

    # Translation Generator
    if self.t > 0:
      # Prepare list of random tranlations given distribution (Uniform)
      if self.rotation_type == "normal":
        t_list = np.random.normal(self.t, self.rotation_std, self.size) #only one direction
        z_list = np.zeros(self.size)
        t_list = np.stack([t_list,z_list], axis=-1) #only one direction
      if self.rotation_type == "uniform":
         if self.rotation_std == 0:
            t_list = np.random.uniform(-1/2*np.abs(self.t), 1/2*np.abs(self.t), (self.size,2)) 
         elif self.rotation_std == 1:
           if self.t>=0:
              t_list = np.random.uniform(0, self.t, self.size)
           elif self.t<0:
              t_list = np.random.uniform(self.t, 0, self.size)
         elif self.rotation_std == -1:
            t_list = np.random.uniform(-1/4*np.abs(self.t), 3/4*np.abs(self.t), (self.size,2)) 
      for img in range(self.size):
        t_i = t_list[img]
        padded_data[img] = ndimage.shift(padded_data[img], t_i)

    

    
    self.data = np.reshape(padded_data, (self.size, 784))
    self.label = np.reshape(padded_label, (self.size, 784))
    self.r_label = r_list
    self.t_label = t_list[:,0] #only one direction
    self.s_label = s_list

  def __len__(self):
    # number of data points
    return self.size

  def __getitem__(self, idx):
    data_point = self.data[idx]
    data_label =  self.label[idx]
    if self.s != 1:
      data_aux = self.s_label[idx]
    if self.r != 0:
      data_aux = self.r_label[idx]
    if self.t != 0:
      data_aux = self.t_label[idx]
    return data_point, data_label, data_aux

def cv2_clipped_zoom(img, zoom_factor):
    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of 
    the image without changing dimensions
    Args:
        img : Image array
        zoom_factor : amount of zoom as a ratio (0 to Inf)
    """
    height, width = img.shape[:2] # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1,x1,y2,x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)
    
    result = cropped_img
    if jnp.shape(cropped_img) != (28,28):
      print(resize_width, resize_height,jnp.shape(cropped_img))
      result = cv2.resize(cropped_img, (resize_width, resize_height))
      result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width
    return result

def show_img_single(img, title='title', name=None, vmin=0., vmax=256.):

    plt.figure(figsize=(5,5))
    if title!='title':
        plt.title(title)
    plt.imshow(img, cmap='gray', vmax=vmax, vmin=vmin)
    if name!=None:
        plt.savefig(name)
    plt.show()

