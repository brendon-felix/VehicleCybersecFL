# Federated Learning for Vehicle Cybersecurity
 
 <img src="https://user-images.githubusercontent.com/124619546/229372532-9bf23b02-e025-4eb9-9bc6-5ebbf8fed500.jpg" width="50%" height="50%">

Federated Learning for Vehicle Cybersecurity is a Colorado State University ECE Senior Design project in which we implement a new federated learning model for vehicle cybersecurity that can be efficiently deployed and securely improved using online distributed training. This model will run in real time to detect intrusions into the Controller Area Network (CAN bus), which is a commonly used network for communication between modern automobile components.

Federated learning (also referred to as collaborative learning) is a method of developing a machine learning model which uses a network of independent devices that train models in parallel and aggregate their results into a single global model. You can view the orginal paper on federated learning [here](https://arxiv.org/abs/1602.05629). We use this technique in our project to determine its feasibility in the context of vehicle cybersecurity by comparing performance between federated and non-federated versions of the same anomaly detector. The recurrent-autoencoder used for our model is derived from INDRA, a recent architecture developed by the [EPiC Lab](http://epic-lab.engr.colostate.edu/) at CSU. The INDRA paper can be found [here](https://ieeexplore.ieee.org/document/9211565).

Vist our project site to learn more: (https://projects-web.engr.colostate.edu/ece-sr-design/AY22/edge/)

The code in this repository includes implementations for both centralized (non-federated) and federated anomaly detectors which are trained and evaluated using the [SynCAN dataset](https://github.com/etas/SynCAN) by ETAS. The code is written in Tensorflow v2 and tested on Google Colab.
