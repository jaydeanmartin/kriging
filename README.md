# kriging
The primary purpose of this repository is to provide public access to algorithms I have written over the years to fit Kriging models to a set of observations.
A Kriging model is a Gaussian spatial process model that is able to reproduce relatively complex behavior due to its form. In most cases it is used as an 
interpolating model for non-lattice distributed observations. 

The majority of the code lies in the kriging directory in specifically in the kriging.py file. The main.py file in the same directory gives an example of reading
csv data for the observations and then creating a kriging model. Kriging is a class tha thas many properties once initialized (created) that can be used for prediction
and uncertainty estimation.

In this root directery are example of Jupyter notebooks that also use the Kriging model class. I would recommend using the Jupyter notebook interface. If you are 
looking at this repository, you most likely already know how to use Jupyter notebooks for data analysis.

I will try to add resources to this repository including some of my publications. I have tried to include comments in most of the code, but if you have questions
feel free to contact me at jaydeanmartin@gmail.com.

