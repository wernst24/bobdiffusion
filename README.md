# bobtextureanalysis
This is a web app for analyzing the fiber structure of images, intended to be used with drumheads. In order to use the application, go to bobtextureanalysis.streamlit.app. Currently, the app is very inefficient and not well designed, so please keep file submissions small. After uploading an image, click "Analyze", and four images should appear in the right column: the original, HSV encoded orientation & coherence, coherence, and orientation.

# Coherence and angle calculation explained
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