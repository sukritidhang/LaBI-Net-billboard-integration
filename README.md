# LaBI-Net-billboard-integration

This code is only  for academic and research purposes.


## Code Organization

All codes are written in python3.


### Dependencies 

The following libraries should be installed before the execution of the codes



- numpy: pip install numpy
- pandas: pip install pandas
- matplotlib: pip install matplotlib
- glob: pip install glob
- scikit-image: pip install scikit-image


### Data

<p>The billboard dataset in this work is the ALOS dataset[1], that stands for
Advert Localization in Outdoor Scenes.  </p>

> Dev, S., Hossari, M., Nicholson, M., McCabe, K., Nautiyal, A., Conran,
C., Tang, J., Xu, W., Pitie, F.: The ALOS dataset for advert localization
in outdoor scenes. In: Proc. Eleventh International Conference on Quality
of Multimedia Experience (QoMEX) (2019)


### Scripts 

- alpha_affine_blending.py : Run this script to blend billboard using alpha blending technique.
- cutpaste_affine_blending.py : Run this script to blend billboard using direct copy and paste method
- poisson_affine_blending2.py : Run this script to blend billboard using poisson equations.
- poisson_lap_affine_blending.py : Run this script to blend billboard using proposed laplace affine blending method.