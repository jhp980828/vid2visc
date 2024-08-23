# vid2visc
### TITLE: Fluid Viscosity Prediction Leveraging Computer Vision and Robot Interaction
#### AUTHORS: Jong Hoon Park, Gauri Pramod Dalwankar, Alison Bartsch, Abraham George, and Amir Barati Farimani

https://www.sciencedirect.com/science/article/pii/S0952197624007619

#### Abstract
Accurately determining fluid viscosity is crucial for various industrial and scientific applications. Traditional methods of viscosity measurement, though reliable, often require manual intervention and cannot easily adapt to real-time monitoring. With advancements in machine learning and computer vision, this work explores the feasibility of predicting fluid viscosity by analyzing fluid oscillations captured in video data. The pipeline employs a 3D convolutional autoencoder pretrained in a self-supervised manner to extract and learn features from semantic segmentation masks of oscillating fluids. Then, the latent representations of the input data, produced from the pretrained autoencoder, are processed with a distinct inference head to infer either the fluid category (classification) or the fluid viscosity (regression) in a time-resolved manner. When the latent representations generated by the pre-trained autoencoder are used for classification, the system achieves a 97.1% accuracy across a total of 4140 test datapoints. Similarly, for regression tasks, employing an additional fully-connected network as a regression head allows the pipeline to achieve a mean absolute error of 0.258 cP over 4416 test datapoints. This study represents an innovative contribution to both fluid characterization and the evolving landscape of Artificial Intelligence, demonstrating the potential of deep learning in achieving near real-time viscosity estimation and addressing practical challenges in fluid dynamics through the analysis of video data capturing oscillating fluid dynamics.
