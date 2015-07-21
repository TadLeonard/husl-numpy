# HUSL color space conversion
A color space conversion library that works with `numpy` arrays. See [www.husl-colors.org](www.husl-colors.org) for more information about the HUSL color space.


## Example 1: Highlighting bluish regions
Let's say we need to highlight the bluish regions in this image:

![an image](images/gelface.jpg)

We'll read the image into a `numpy.ndarray` and proceed from there.

```python
import imread  # a great library for reading images as numpy arrays
import nphusl 

# read in an ndarray of uint8 RGB values
img = imread.imread("images/gelface.jpg")
out = img.copy()  # the array we'll modify in the next examples
```

#### Example 1A: With the HUSL color space

The HUSL color space makes this pretty easy. Blue hues are roughly between
250 and 290 in HUSL.

```python
# make a transformed copy of the image array
hue = nphusl.to_hue(img)  # a 2D array of HUSL hue values
bluish = np.logical_and(hue > 250, hue < 290)  # create a mask for bluish pixels
out[~bluish] *= 0.5  # non-bluish pixels darkened
```

At this point, the `out` image looks like what we'd expect:

![this image](images/blue.jpg)


#### Example 1B: With the RGB color space

With the RGB color space, we have to examine each color channel and select
pixels that match three conditions:

1. a pixel's blue channel must be greater than its red channel
2. a pixel's blue channel must be greater than its green channel
3. a pixel's blue channel must be greater than some arbitrary number
   (depending on just *how blue* we want our selection to be)

```python
R, G, B = (img[..., n] for n in range(3))  # break out RGB color channels

# we'll try to create a bluish selection by choosing pixels for which
# the blue channel has a greater value than the others
bluish = np.logical_and(B > R, B > G)  # no overpowering red or green
bluish = np.logical_and(bluish, B > 125)  # strong enough blue channel
out[~bluish] *= 0.5  # non-bluish pixels darkened
```

Again, we get approximately what we hoped for:

![this image](images/blue_rgb.jpg)

Working with plain red, green, or blue doesn't present a very challenging
problem, but the clarity of having hue on its own separate channel is
still apparent. The separation of hue and and lightness into distinct channels
can lend even more flexibility. The next examples illustrate this.


## Example 2: Highlighting bright regions

This example shows the ease of selecting pixels based on perceived
"luminance" or "lightness" with HUSL.

```python
hsl = nphusl.to_husl(img)  # a 3D array of HUSL hue, saturation, and lightness
lightness = hsl[..., 2]  # just the lightness channel
dark = lightness < 62  # a simple choice, since lightness is in (0..100)
out[dark] = 0x00  # change selection to black
```

This code gives us the light regions of the subject's face with a
black background:

![this image](images/light.jpg)


## Example 3: Melonize

As a completely arbitrary challenge, let's highlight small changes in hue.
We'll walk along the HUSL hue spectrum in steps of 5 (the HUSL hue range
runs from 0 to 360). As we walk through each hue range, we'll alternate our
effect on the image's pixels to create green and pink striations -- a
kind of "watermelon" effect.

```python
hsl = nphusl.to_husl(img)
hue, _, lightness = (hsl[..., n] for n in range(3))
pink =  1.3, 0.3, 0.6 
green = 0.3, 1.1, 0.3
for low, high in nphusl.chunk(360, 5):  # chunks of the hue range
    select = np.logical_and(hue > low, hue < high)
    is_odd = low % 10
    color = pink if is_odd else green
    out[select] *= color
```

This code gives us a nicely melonized face:

![this image](images/watermelon_flat.jpg)

One thing I don't like about this is that the image looks somewhat flat.
This is because our transormation focused only on *hue*. The light/dark
regions give the image depth. We can exaggerate depth by using
our HUSL lightness value as a multiplier.

```python
for low, high in nphusl.chunk(100, 10):  # chunks of the lightness range
    select = np.logical_and(lightness > low, lightness < high)
    out[select] *= (high / 100.0)
```

That gives us the same melonized subject, but with dark regions that
receed into the background dramatically:

![this image](images/watermelon.jpg)




