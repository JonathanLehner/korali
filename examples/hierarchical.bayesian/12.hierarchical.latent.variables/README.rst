.. _hsaem_examples:

===============================================================================
Optimizing parameters in hierarchical latent variable models, using SA-EM
===============================================================================

Here, we want to find the hyper-parameters maximizing the probability of some datapoints. The probability function
additionally depends on latent variables, and as in the latent variable example in `examples/bayesian.inference/latent`,
we will use a variant of the expectation maximization algorithm to solve the problem.

In contrast to the other example, here we assume that our problem has a hierarchical structure, and thus can use a
more specialized solver, `Hierarchical Stochastic Approximation Expectation Maximization (HSAEM)`. This solver simplifies
setting up the problem, because it internally handles sampling and one level of the two-level hierarchical probability
model.



Problem setting
-------------------
As in `examples/bayesian.inference/latent`, we aim to find the parameters for a
probability distribution that maximize the likelihood of a set of observed points :math:`X = \{x_i\}_i`,
:math:`x_i \in \mathbb{R}^n`. For comparison, we first use the standard version of the `SA-EM` solver for one problem,
and then show the use of specialized `SA-EM` for hierarchical models (`HSAEM`), for the same problem as well as three others.

As hierarchical model, we consider a probability distribution of the form (following [1]):


.. math::
    p(x_i, \theta_i | \psi)  = p( x_i | \theta_i ) \cdot p( \theta_i | \psi)

Here, each i indexes one 'individuum' from a population. Multiple datapoints might exist for a single individuum.

(Any individual covariates need to be absorbed into the model :math:`p( x_i | \theta_i )`.)


**Structured prior assumption:**

We assume the latent variables :math:`\theta_i` can be transformed
into (or originate from) normally distributed variables :math:`z_i = h(\theta_i)`, i.e.

.. math::

    x_i \; \sim \;  p(x_i | z_i)  \\
    z_i \; \sim \; \mathcal{N}(\beta, \Omega)

Here, :math:`\mathcal{N}(\beta, \Omega)` is the normal distribution with mean  :math:`\beta` and covariance matrix :math:`\Omega`.

To this end, the user can choose between latent variable coordinates :math:`\theta_{i,j}` following a normal, log-normal or logit-normal
distribution. Log-normal refers to :math:`log(\theta_{i,j}) \; \sim \; \mathcal{N}(a, b)`, and logit-normal similarly means
:math:`logit(\theta_{i,j}) \; \sim \; \mathcal{N}(a, b)`, where :math:`logit(\theta) = log\left({\theta}/{(1 - \theta)}\right)`.
Normal and log-normal distributions describe distributions in many different disciplines `[2] <https://stat.ethz.ch/~stahel/lognormal/bioscience.pdf>`_.

1. **Simple population example:**
   We first follow the simple example of section 9.2.5 of Lavielle's book [1]. All distributions are normal which makes
   sampling distribution and optimization analytically solvable [1].
   All variables are scalar. The model is:

   .. math::

      x_i \; \sim \; \mathcal{N}(\theta_i, \sigma^2) \; \text{for} \; 1 \leq i \leq N  \\
      \theta_i \; \sim \; \mathcal{N}(\psi, \omega^2)

   We have one sample per 'individuum' i. Note that our notation is flipped compared to the book, we use :math:`\theta`
   for latent variables and :math:`\psi` for hyperparameters.

   We solve this problem both with SAEM (`run-saem-population-simple.py`) and with HSAEM (`run-saem-hierarchical.py`).


2. **Mixed prior distributions:**
    Here we extend the previous example to
     -  arbitrary numbers of dimensions
     -  all three types of prior distributions: Normal, log-normal, and logit-normal
     -  different numbers of measurements for each individual.

    The resulting probability distribution is:

   .. math::

      \mathbf{x}_{ij} \; \sim \; \mathcal{N}(\mathbf{\theta}_i, \;\mathbf{I}\cdot\sigma^2) \;\; \text{for} \;\; 1 \leq i \leq N, \; 1 \leq j \leq n_i  \\
      \mathbf{\theta}_i = \mathbf{f}(\mathbf{z}_i) \\
      \mathbf{z}_i \; \sim \; \mathcal{N}(\mathbf{\beta}, \mathbf{\Omega})

   That is, the data is again normally distributed around each individual's latent variable :math:`\mathbf{\theta}_i`, which is
   now vector-valued and whose transformed entries follow a multivariate normal distribution with mean :math:`\mathbf{\beta}`
   and covariance matrix :math:`\mathbf{\mathbf{\Omega}}`.
   The inverse transform :math:`\mathbf{f}` is the identity in each normal coordinate of :math:`\mathbf{\theta}_i`, while
   for log-normal coordinates it's the exponential and for logit-normal coordinates the sigmoidal function:

   :math:`\mathbf{f}(\mathbf{z})_k = f(z_k)`, with:

   - :math:`f(z_k) = z_k` for normal :math:`\theta_k`,
   - :math:`f(z_k) = exp(z_k)` for log-normal :math:`\theta_k`, and
   - :math:`f(z_k) = \frac{1}{1 + exp(-z_k)}` for logit-normal :math:`\theta_k` (sigmoid is the inverse logit).


Data for both examples can be generated with custom parameters using the script
`_model/[example name]/generate_data.py` (please change the parameters in the script).
This will also write more detailed information about the example into `_model/[example name]/data_[name]_info.txt`. Please
refer to this for an estimate how difficult the generated data will be to fit - it generates
a number of proxies for the optimizers. These can be compared to the true / original parameter values. For example,
accuracy for logit-normal variables is affected strongly by the variance :math:`\sigma^2`, since their values
are restricted to the unit interval.

The two previous examples showed different models for the `prior` :math:`p(\theta_i | \beta, \Omega)`.
We now look at more complex `conditional probability` models :math:`p(x_i | z_i)`.

3. **"Normal" example:**
   We now consider data points :math:`y_i \in \mathbb{R}` describing a functional relationship:

   .. math::

      y_i = f( \theta_{i,1}) + \epsilon =  \theta_{i,1}^2 + \epsilon\\
      \epsilon \sim N(0, \theta_{i,2}^2)

   Here we have a four-dimensional latent space, :math:`\mathbf{\theta_i} = (\theta_{i,1}, \theta_{i,2})`, one
   describing the noise :math:`\epsilon`, the other with functional relationship to data :math:`y_i`. The functional
   dependence is nonlinear and not one-to-one: For vanishing noise, there are two possible :math:`\theta_{i,1}` for
   each :math:`y_i`.

   There are :math:`N = 200` individuals with one measurement :math:`y_i` each. Data can be regenerated by running
   `_data/normal/generate_data.py`.

   **Model:**
   Contrary to the data generation process, we will assume in `run-saem-normal.py` that :math:`\theta_{i,1}`
   and :math:`\theta_{i,2}` follow log-normal distributions.

   **True parameter values:**

   - :math:`\theta_{i,1} \sim N(5, 0.1)`
   - :math:`\theta_{i,2} = 1`


4. **Logistic example:**
   We again consider data points :math:`(x_i, y_i) \in \mathbb{R}^2` originating from a functional relationship:

   .. math::

      y_i = f( x_i, \theta_{i,1}, \theta_{i,2}, \theta_{i,3}) + \epsilon, \;\;\;with \\
      f(x, \theta_1, \theta_2, \theta_3) = \frac{\theta_1 \theta_2 \cdot e^{x\cdot\theta_3} }{\theta_1 + \theta_2\cdot(e^{x\cdot\theta_3} - 1)}, \;\;\;and\\
      \epsilon \sim N(0, \theta_{i,4}^2).

   f is a parameterized logistic function, and :math:`theta_{i,4}` again determines the noise variance.

   There are :math:`N = 10` individuals with multiple measurements for :math:`x_{i,k} \in [0, 10]` and corresponding
   :math:`y_{i,k}`. Data can be regenerated by running `_data/logistic/generate_data.py`.

   **Model:**
   We will assume a normal distribution for each of :math:`\theta_{i,1}, \theta_{i,2}` and :math:`\theta_{i,3}`, and
   a log-normal distribution for :math:`\theta_{i,3}` in `run-saem-logistic.py`.

   **True parameter values:**

   - :math:`\theta_{i, 1} \sim N(200, 20)`
   - :math:`\theta_{i, 2} \sim N(40, 10)`
   - :math:`\theta_{i, 3} \sim N(1, 0.1)`
   - :math:`\theta_{i, 4} = 5`



File descriptions
------------------

- The scripts :code:`run-saem-[example name].py` run the five different examples (for the simple example there are two:
  :code:`run-saem-hierarchical.py` uses HSAEM, :code:`run-saem-population-simple.py` uses SAEM.)

..
   - :code:`test-saem-normal.py`: Not included in master

- Subfolder :code:`_model/` contains code for the probability models for each example. It accesses the data in
- subfolder :code:`_data/[example name]/`, where also the data-generating scripts can be found.


How to run the examples
--------------------------------------------

In a shell from this directory, type, for example

.. code-block::

    $ python3 run-saem-hierarchical.py

to run the 'simple example' using HSAEM.





Simple example using standard SA-EM
--------------------------------------------

This example is structured similarly to the examples in `these examples <../../bayesian.inference/latent/README.rst>`_.
. Please refer to their readme
for explanations.

Simple example using HSAEM
---------------------------


We first import everything from the file :code:`_model/simple_example_Lavielle/model.py`, including
our model class, :code:`ConditionalDistribution4`. We also import the :code:`korali` Python library:
(The distribution enumeration continues from the non-hierarchical examples:)

.. code-block:: python

    import sys
    sys.path.append('./_model/simple_example_Lavielle')
    from model import *

    import korali

We then instatiate the model class, which provides acces to the data points and the conditional distribution function:

.. code-block:: python

    distrib = ConditionalDistribution4()

To run a Korali experiment, we first need to create a :code:`korali.Experiment` that we can then customize.
We will also need a :code:`korali.Engine` to run the experiment:

.. code-block:: python

    k = korali.Engine()
    e = korali.Experiment()

**Problem Setup:**
To solve a hierarchical problem with latent variables, we tell Korali that :code:`HierarchicalLatent` is the
problem type. We then set the conditional log likelihood function, i.e. :math:`p(x | \theta)`

.. code-block:: python

    e["Problem"]["Type"] = "Bayesian/Latent/HierarchicalLatent"
    e["Problem"]["Conditional Log Likelihood Function"] = lambda sample : distrib.conditional_p(sample)

To evaluate total and inidividual likelihoods, the problem needs access to our measured or generated
data points. We set them as the problem's :code:`"Data"`, and also define the number of dimensions
and individuals of our problem:

.. code-block:: python

    # We need to add one dimension to _p.data, because one individual in the general case could have
    # more than one data point assigned
    data_vector = [[] for _ in range(distrib._p.nIndividuals)]
    for i in range(distrib._p.nIndividuals):
        data_vector[i].append([distrib._p.data[i]])
    e["Problem"]["Data"] = data_vector
    e["Problem"]["Data Dimensions"] = 1
    e["Problem"]["Number Individuals"] = distrib._p.nIndividuals
    e["Problem"]["Latent Space Dimensions"] = 1

**Solver Setup:** We then define the solver. We want to use :code:`HSAEM`. We can also pass additional parameters for the solver.
If they are not passed, default values will be used. Here, we choose to use a short sampling process with
5 chains, only one main sampling step with 6 sub-steps (N1 + N2 + N3). Finally, we want to run HSAEM for
30 generations:

.. code-block:: python

    e["Solver"]["Type"] = "HSAEM"
    e["Solver"]["Number Samples Per Step"] = 5
    e["Solver"]["mcmc Outer Steps"] = 1
    e["Solver"]["N1"] = 2
    e["Solver"]["N2"] = 2
    e["Solver"]["N3"] = 2
    e["Solver"]["Termination Criteria"]["Max Generations"] = 30

**Variables and Distributions:**
Apart from solver and problem, we define what variables our experiment has. Each variable also needs
a prior distribution, since our selected problem :code:`HierarchicalLatent` is a :code:`Bayesian` problem:

.. code-block:: python

    e["Distributions"][0]["Name"] = "Uniform 0"
    e["Distributions"][0]["Type"] = "Univariate/Uniform"
    e["Distributions"][0]["Minimum"] = -100
    e["Distributions"][0]["Maximum"] = 100

Instead of defining each latent variable for each individual, problem :code:`HierarchicalLatent` allows
us to only define the latent variables for one individual as prototypes. There is only
one latent space dimension in our problem, so we define only one latent variable. Latent variables for
all other individuals, as well as hyperparameters, will be automatically inferred and added by Korali.

.. code-block:: python

    e["Variables"][0]["Name"] = "latent mean "+str(0)
    e["Variables"][0]["Initial Value"] = -5
    e["Variables"][0]["Latent Variable Distribution Type"] = "Normal"
    e["Variables"][0]["Prior Distribution"] = "Uniform 0"

Here, we gave each variable a name for identification and set a starting value (this will be used to set the
starting value for the hyperparameter representing the mean for this latent variable).
The field :code:`"Latent Variable Distribution Type"` defines how we expect this variable to be distributed.
It can be one of :code:`"Normal"`, :code:`"Log-Normal"` and :code:`"Logit-Normal"`.

Finally, we choose to only store the experiment state every 50 generations (for plotting, a frequency of 1
is advised) and change the default results folder. We also tell Korali to print :code:`"Detailed"`
information to the command line every 10 generations:


.. code-block:: python

    e["File Output"]["Frequency"] = 50
    e["File Output"]["Path"] = "_korali_result_hierarchical/"
    e["Console Output"]["Frequency"] = 10
    e["Console Output"]["Verbosity"] = "Detailed"

Now we can run the experiment and wait for the results.

.. code-block:: python

    k.run(e)





References
==========

[1] Lavielle, Marc. Mixed effects models for the population approach: models, tasks, methods and tools. CRC press, 2014.

`[2] <https://stat.ethz.ch/~stahel/lognormal/bioscience.pdf>`_ Limpert, Stahel, Abbt. Log-normal Distributions across the Sciences: Keys and Clues. BioScience May 2001.

