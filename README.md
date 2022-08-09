# Array pattern calculator

## By Gerson Villalba Arana


## General description

Array pattern calculator for arrays of radiant elements and some plotting functions. 

* Each individual radiant element can be placed in any spacial position (x, y, z).
* Each individual radiant element can be powered with a specific amplitude (in power) and phase (specified in degrees).
* The feed amplitudes will be normalized, so that the total will always be 1 (0dB).
* The resulting calculated Gain will be expressed in dB.
* It allows to load a specific element pattern.
* If you don't specify a element pattern, a cosine power will be used. The default value for the power is 1.8, which gives a symmetrical 68 degrees pattern.


## Example of use

Jupyter notebook file "arrayExample" contains some examples of use of the package.


## To be done:

* Add frequency vector as an attribute to the array class to compute multiple frequencies calculations at once.
* Add multiple element radiation patterns to the array, one for each frequency.

## License

See license.txt file
