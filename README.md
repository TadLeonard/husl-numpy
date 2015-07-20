# numpy-friendly HUSL color space conversions

A color space conversion library that works with `numpy` arrays. See [www.husl-colors.org](www.husl-colors.org) for more information about the HUSL color space.


Some images
^^^^^^^^^^^^^
Given an image like ![this one](images/gelface.jpg), let's say we want to
highlight the bluish regions of the image.


With the HUSL color space
-------------------------
The HUSL color space makes this pretty easy. Blue hues are roughly between
250 and 290 in HUSL.

```python
import imread  # a great library for reading images as numpy arrays
import nphusl 

# read in an ndarray of uint8 RGB values
img = imread.imread("images/gelface.jpg")

# make a transformed copy of the image array
out = img.copy()
hue = nphusl.to_hue(img)  # a 2D array of HUSL hue values
bluish = np.logical_and(hue > 250, hue < 290)  # create a mask for bluish pixels
out[bluish] = (0, 0, 255)  # highlight bluish area bright blue

# write modified image to disk
imread.imwrite("blue.jpg", out)
```

This results in ![this image](images/blue.jpg)


With the RGB color space
------------------------
With the RGB color space, we have to examine each color channel and select
pixels that match three conditions:

1. a pixel's blue channel must be greater than its red channel
2. a pixel's blue channel must be greater than its green channel
3. a pixel's blue channel must be greater than some arbitrary number
   (depending on just *how blue* we want our pixels to be)

```python
out = img.copy()
R, G, B = (img[..., n] for n in range(3))  # break out RGB color channels

# we'll try to create a bluish selection by choosing pixels for which
# the blue channel has a greater value than the others
select = np.logical_and(B > R, B > G)  # no overpowering red or green
select = np.logical_and(select, B > 125)  # strong enough blue channel
out[select] = (0, 0, 255)  # reveal the selected region

# write the modified image to disk
imread.imwrite("blue_rgb.jpg", out)
```

This results in ![this image](images/blue_rgb.jpg)



