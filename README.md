# imageSeg
Image segmentation project for ACM 270-02 Course Project 

This code implements and tests an image segmentation algorithm to identify clusters in images. The algorithm is based on 
a spectral clustering method using continuum Laplacians. 

- To run the code on a new "real-world" image, call run_segmentation with the necessary inputs. Pass in the relative 
path to the image in addition to the pde parameters. An example of this is provided in the real_image_example.m file. The 
resulting segmentation figures will be produced in the res_images/filename folder (filename should be specified 
when calling run_segmentation).

- To run the code to segment a synthetic image, call run_segmentation with the necessary inputs. Pass in the synthetic 
image density and a ground truth matrix in addition to the pde parameters. An example of this is provided in the 
synthetic_image_example.m file. The resulting segmentation figures will be produced in the res_images/filename folder 
(filename is specified when calling run_segmentation).

- To run an experiment with varying (p, q, r) values as discussed in our report, run one of the following commands. The 
results will appeare in the res_image/experiments folder under the experiment title (balanced, smallq, or bigq). The 
results are grouped in folders each image. Each folder corresponding to an image has sub-folders corresponding to 
each (p, q, r) value 
  - experiment_balanced.m for the balanced case
  - experiment_smallq.m for the q < p + r case
  - experiment_bigq.m for the q > p + r case

Pre-run experiment figures are available at https://caltech.box.com/s/2nysvxi7n9tu8azcl6d441sr9co0kzlz 
