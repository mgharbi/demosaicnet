"""Utilities to make a mosaic mask and apply it to an image."""
import numpy as np
import torch as th


__all__ = ["bayer", "xtrans"]


def bayer(im, return_mask=False):
  """Bayer mosaic.

  The patterned assumed is::

    G r
    b G

  Args:
    im (np.array): image to mosaic. Dimensions are [c, h, w]
    return_mask (bool): if true return the binary mosaic mask, instead of the mosaic image.

  Returns:
    np.array: mosaicked image (if return_mask==False), or binary mask if (return_mask==True)
  """

  numpy = False
  if type(im) == np.ndarray:
    numpy = True

  if type(im) == np.ndarray:
    mask = np.ones_like(im)
  else:
    mask = th.ones_like(im)

  # red
  mask[..., 0, ::2, 0::2] = 0
  mask[..., 0, 1::2, :] = 0

  # green
  mask[..., 1, ::2, 1::2] = 0
  mask[..., 1, 1::2, ::2] = 0

  # blue
  mask[..., 2, 0::2, :] = 0
  mask[..., 2, 1::2, 1::2] = 0

  if not numpy:  # make it a constant for ONNX conversion
    mask = th.from_numpy(mask.cpu().detach().numpy()).to(im.device)

  if mask.shape[0] == 1:
    mask = mask.squeeze(0) # coreml hack

  if return_mask:
    return mask

  return im*mask


def xtrans_cell(torch=False):
  g_pos = [(0,0),        (0,2), (0,3),        (0,5),
                  (1,1),               (1,4),
           (2,0),        (2,2), (2,3),        (2,5),
           (3,0),        (3,2), (3,3),        (3,5),
                  (4,1),               (4,4),
           (5,0),        (5,2), (5,3),        (5,5)]
  r_pos = [(0,4),
           (1,0), (1,2),
           (2,4),
           (3,1),
           (4,3), (4,5),
           (5,1)]
  b_pos = [(0,1),
           (1,3), (1,5),
           (2,1),
           (3,4),
           (4,0), (4,2),
           (5,4)]

  if torch:
    mask = th.zeros(3, 6, 6)
  else:
    mask = np.zeros((3, 6, 6), dtype=np.float32)

  for idx, coord in enumerate([r_pos, g_pos, b_pos]):
    for y, x in coord:
      mask[..., idx, y, x] = 1

  return mask

def xtrans(im, return_mask=False):
  """XTrans Mosaick.

   The patterned assumed is::

     G b G G r G
     r G r b G b
     G b G G r G
     G r G G b G
     b G b r G r
     G r G G b G

  Args:
    im(np.array, th.Tensor): image to mosaic. Dimensions are [c, h, w]
    mask(bool): if true return the binary mosaic mask, instead of the mosaic image.

  Returns:
    np.array: mosaicked image (if mask==False), or binary mask if (mask==True)
  """

  numpy = False
  if type(im) == np.ndarray:
    numpy = True
    mask = xtrans_cell(torch=False)
    # mask = np.zeros((3, 6, 6), dtype=np.float32)
  else:
    # mask = th.zeros(3, 6, 6).to(im.device)
    mask = xtrans_cell(torch=True).to(im.device)
    if len(im.shape) == 4:
      mask = mask.unsqueeze(0).repeat(im.shape[0], 1, 1, 1)

  h, w = im.shape[-2:]
  h = int(h)
  w = int(w)

  new_sz = [np.ceil(h / 6).astype(np.int32), np.ceil(w / 6).astype(np.int32)]

  sz = np.array(mask.shape)
  sz[:-2] = 1
  sz[-2:] = new_sz
  sz = list(sz)

  if numpy:
    mask = np.tile(mask, sz)
  else:
    mask = mask.repeat(*sz)

  if return_mask:
    return mask

  return mask*im
