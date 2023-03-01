# Federated Learning for Vehicle Cybersecurity
 
![FL_IDS_final](https://user-images.githubusercontent.com/124619546/217645343-32de26d5-b91f-4ad5-8b27-c89569b7de6a.png)

Federated Learning for Vehicle Cybersecurity is a Colorado State University senior design engineering project in which we implement a new federated learning model for vehicle cybersecurity that can be efficiently deployed and securely improved using online distributed training. This model will run in real time to detect intrusions into the Controller Area Network (CAN bus), which is a commonly used network for communication between an automobile’s various microcontrollers.

Federated learning (also referred to as collaborative learning) is a method of developing a machine learning model which uses a network of independent edge devices that train in parallel and distribute results to other devices. You can view the orginal paper on federated learning by *McMahan et al.* [here](https://arxiv.org/abs/1602.05629). We will use this technique as a proof of concept to train our cybersecurity model more effectively than we could with a single device while preserving data privacy for any future user base.

Vist our project site to learn more: (https://projects-web.engr.colostate.edu/ece-sr-design/AY22/edge/)

The code in this repository includes implementations of both centralized and federated anomaly detectors trained using the [SynCAN dataset](https://github.com/etas/SynCAN). The code is written in Tensorflow v2 and tested on Google Colab.
