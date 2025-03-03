# notes before review
Priorities:
1. Change dtype of calculation to np.float16 (or parameterized)
2. Add scale invariance support for inner sigma & epsilon
3. Find desired export format for coherence/angle images - is uint8 enough?

Current issues:
1. type upcasting is really annoying
2. not sure what epsilon should be proportional to - it's added to eigenvalues, so should have same units - which will be same units of structure tensor. This is the image gradient squared, so like (intensity / position)^2 ? I have no clue, need to think about this more

# notes after review