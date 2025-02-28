# bobtextureanalysis
This is a web app for analyzing the fiber structure of images, intended to be used with drumheads, developed for RIT FIP 2024-25. The app can be hosted locally by installing `streamlit`, or used through the web app at https://bobdiffusion.streamlit.app.

# parameters
1. **Downscale**
    This is the factor by which the dimensions of the input image will be multiplied by, for an interpolated downscale. Leave 1 for no downscale.

2. **Coherence gamma**
    This is the exponent for $coherence_{scaled} = coherence^{\gamma}$. This allows for emphasis on larger coherence values.

3. **Histogram blur sigma**
    This is the standard deviation for a gaussian blur on the histogram - leave 0 for no blur

4. **Sigma to ydim ratio**
    This is used for the "outer scale" of structure tensor calculation - specifically used for the standard deviation of image gradient calculation via Derivative of Gaussian filter. If the value chosen results in a stdev below ~.5px, the processing will not work as intended.
    If an outer scale in units of length is desired, use the formula: $$Ratio_\sigma = \frac{\sigma \,\text{[length]}}{y \,\text{[length]}}$$

5. **Inner sigma**
    This is the standard deviation of the "inner scale" blur, which allows for a selection of desired spatial frequency range. Smaller values will keep high-frequency information, while larger values will shift emphasis to lower-frequency information.

6. **epsilon**
    This is used during the coherence calculation: $$coherence = (\frac{\lambda_1 + \lambda_2}{\lambda_1 - \lambda_2 + \epsilon})^2$$ Values near zero result in numerical instabilities for near-constant regions, where both eigenvectors of the structure tensor are small. Additionally, unusually large values (eg. 0.01) may result in a desirable attenuation of low-coherence regions for fiber analysis.

7. **num_bins**
    This is the number of bins for the coherence-weighted angle histogram, more bins will result in a higher-resolution histogram, while less bins will speed up compute time.

<!-- # Coherence and angle calculation explained
1. calculate x and y gradient with gradient filter - add menu to choose type of computation and sigma (if applicable)

2. calculate structure tensor from x and y gradient
(J = [[I_x ** 2, I_x * I_y], [I_x * I_y, I_y ** I_y]] = [[mu_20, mu_11], [mu_11, mu_02]])
$$ m $$

3. calculate k20 and k11, which fully describe structure tensor
k20 = mu20 - mu02 + 2i*mu11 = (lambda1 - lambda2)exp(2i*phi)
k11 = mu20 + mu02 = lambda1 + lambda2 (trace of matrix = sum of eigenvectors)

4. blur the k values with gaussian to select frequency bandwidth for angles

5. calculate coherence and angle from k values
|k20|/k11 = sqrt(coherence), and atan2(im(k20), re(k20)) = orientation

The most important parameter (besides the image, silly) is the sigma for the k-value blur.
A low sigma value will put the focus on the angle of higher-frequency patterns, while a higher sigma value will focus on more gross details.

TODO:
- Make session state stuff less jank, decompose more and pass less parameters

- Get a better angle indicator label

- Make site format better (wide?) and reorganize to use more horizontal space. Add options for what output images to show

- And add options for gradient calculation. Add option for gradient calc with larger kernel to avoid artifacts on banded regions - sky, etc.

- Add more explanations, or add link to explanation.

- Add option to show histogram of angles and coherences, and figure out if people in FIP would like other metrics.

- Experiment with crop feature, and create custom convolution function for circular crop, to ignore areas that are blocked out. Add ROI and image explorer feature.

- Experiment with thresholding for different metrics like coherence and angle

- Figure out how to max pool all image layers by only coherence layer - find pixel with max coherence, and pass that entire slice on.

- Do runtime analysis to find bottlenecks

- Find out if streamlit gpu acceleration exists

# Bilateral Filter

Using the coherence calculation already implemented, I want to investigate using coherence in a bilateral filtering implementation. Hopefully this can provide useful regulation for further image processing, or at least look cool.


TODO:
1. make ipynb for testing out bilateral filter
2. make functions in modules.py for arbitrary filters
3. for useful filters, find more efficient implementation -->