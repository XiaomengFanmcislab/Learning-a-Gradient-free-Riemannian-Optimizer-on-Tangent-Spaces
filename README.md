#Learning-a-Gradient-free-Riemannian-Optimizer-on-Tangent-Spaces.
We conduct experiments on three tasks: PCA on the Grassmann manifold, face recognition on the Stiefel manifold, and clustering on the SPD manifold. 

Folders and files
Grassmann_PCA                   -The code of PCA on the Grassmann manifold.
Stiefel_face_recognition        -The code of face recognition on the Stiefel manifold.
SPD_clustering                  -The code of clustering on the SPD manifold.
README.txt                       -This readme file.


Prerequisites
Our code requires PyTorch v1.0 and Python 3.

How to train the optimizers of the PCA task?
1. Go to the folder of the PCA task.
2. Run train.py.


How to test the optimizers of the PCA task?
1. Go to the folder of the PCA task.
2. Change the load path of the trained optimizer in test.py to the path which was setted in the training stage.
3. Set the save path of loss in test.py to where you intend to save.
4. Run test.py.


How to train the optimizers of the face recognition task?
1. Go to the folder of the face recognition task.
2. Run train.py.

How to test the optimizers of the face recognition task?
1. Go to the folder of th face recognition task.
2. Change the load path of the trained optimizer in test.py to the path which was setted in the training stage.
3. Set the save path of loss in test.py to where you intend to save.
4. Run test.py.


How to train the optimizers of the clustering task?
1. Go to the folder of the clustering task.
2. Run train.py.

How to test the optimizers of the clustering task?
1. Go to the folder of the clustering task.
2. Change the load path of the trained optimizer in test.py to the path which was setted in the training stage.
3. Set the save path of loss in test.py to where you intend to save.
4. Run test.py.


