# Welcome to PINACLES Documentation

## A Brief Introduction

 ***Predicting INteractions of Aerosol and Clouds in Large Eddy Simulation*** (PINACLES) is a Python based, massively parallel, anelatic atmospheric model developed at the Pacific Northwest National Laboratory (PNNL) initially to support the large eddy simulation (LES) modeling needs of the Enabling Aerosol-cloud interactions at GLobal convection-permitting scalES (EAGLES) project. PINACLES has grown considerably beyond the EAGLES project into a much more general atmospheric modeling tool. Despite being written in Python, PINACLES maintains performance comparable to models written in compiled languages like Fortran and C++ through [just-in-time (JIT)](https://en.wikipedia.org/wiki/Just-in-time_compilation) compilation provided [Numba](https://numba.pydata.org/) which complies Python code into performant machine code using [LLVM](https://llvm.org/).

## Design Philosophy and Optimization

The traditional software optimization strategy used in atmospheric models seeks to optimize for model throughput. For example, how long in wall clock time it takes a model to simulate one unit of simulation time. PINACLES, takes a different approach, optimizing for what we refer to as ***scientific throughput***. By scientific throughput we mean the rate of scientific discovery per person-year for the average model user. In our minds optimizing for scientific throughput is a two part optimization for: 

1. Computational performance (as in the historical approach)
2. How long does it take for a scientific user to correctly express a new scientific idea or new scientific experiment within a model

Our implementation of this optimization approach is very much inspired by the success that has been found within the machine learning community, in using languages like Python, and to a lesser extent Julia, to make cutting edge research tools and technological advances available to the masses, even those with little software engineering experience, with unprecedented speed and alacrity. In the context of machine learning, this has been referred to by [Chollet](https://dl.acm.org/doi/10.5555/3203489) as the democratization of machine learning. 


