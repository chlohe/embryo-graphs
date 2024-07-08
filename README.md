# Embryo Graphs
This is the official repository for `Embryo Graphs: Predicting Human Embryo Viability from 3D Morphology`.

What's going on here? Well, allow me to explain. We have 4 main folders here.

`cell-segmentation` contains code that does cell segmentation in embryos. You wanna start here.

`embryo-visualiser` contains code that visualises the segmented embryos as 3D models. It also has some handy utilities that allows you to calculate adjacency matrices between embryo cells.

`embryo-graphs` contains code that predicts the clinical outcome of an embryo, based on its graph representation. It also contains a couple other baselines including a bog-standard convolutional neural net.

`via-on-steroids` contains a forked version of the [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/) with some special features for annotating embryo focal stacks.

Each has its own README.
