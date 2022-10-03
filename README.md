# Deep convolutional neural networks for infinite support filtering
My first research project in image processing where I studied Land's kernel $\frac{1}{\sqrt{x^2 + y^2}}$ in order to perform applications on images such as image editing or contrast enhancement.

The files are the following :

UNET.py : the architecture to perform Land's filter with the UNET

Blur.py : functions to apply Land's kernel, Gaussian kernel or the Retinex

Data/train : folder to test performances

save : the latest saves to perform a Gaussian blur with $\sigma = 3$ and $\sigma = 5$.

train : folder to train the U-Net

More informations about this project are available here : https://www.antoinertk.fr/land.
