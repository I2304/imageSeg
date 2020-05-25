# imageSeg
Image segmentation project for ACM 270-02

The function get_segmentation() in the file acm270project.m takes in: 
- path (string): the relative path to an image
- P (float): the value of p to be used in the normalization
- Q (float): the value of q to be used in the normalization
- R (float): the value of r to be used in the normalization 
- K (integer): the number of eigenfunctions (e.g., 6) to be plotted
- maxL (integer): the maximum magnitude eigenvalue (e.g., 40) to be solved for (if maxL is not large enough, an error will be reported; if maxL is too large the code may take a long time to run)
- num_clusters_u (integer): number of clusters when segmenting on u (typically, set this to K+1)
- num_clusters_v (integer): number of clusters when segmenting on v (typically, set this to K+1)
- swap (bool): if the segmentation is bad, consider switching swap (from true to false or vice versa) in order swap the contrast of the image, as this may improve the calculation of the eigenfunctions

The function displays the two segmentations of the image. The first segmentation is done using the Laplacian embedding (based on the eigenfunctions u). The second segmentation is done using the embedding in terms of the transformed eigenfunctions v. In these segmentations, each pixel is labelled with a certain index from 1 to num_clusters_u (or num_clusters_v for the second segmentation). The label idicates the cluster/artifact assignmen of that pixel within the image. 

You may uncomment the examples at the top of acm270project.m to run the algorithm on a few test cases. 

Several examples of the algorithm output, for different values of P, Q, and R are available in the test_cases folder. The test_cases folder has several subfolders, each of which corresponds to a certain test image. In each of these image subfolders, one can find the original image and example segmentations. For the example segmentations, the naming convention is as follows: 
- segmentation_P_Q_R
so that (for exmample)
- segmentation_3/2_2_1/2 
corresponds to segmentation using the normalized continuum Laplacian operator. 
