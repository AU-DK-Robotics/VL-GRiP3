tac1, tac2, and tac3 are the names of the objects you want to register.

The CAD folder is where you need to place your cad.stl file.

Inside each of the tac1, tac2, and tac3 folders, add your mesh.ply file.
tac1 shows how the folder should look after you run build_training_file.py.
tac2 and tac3 should follow the exact same structure.

You should have the following files (see image). We cannot leave our object there because it is an industrial patented component:

tac1.ply (point cloud of your occluded object)

1.txt (the transformation matrix aligning the occluded object with the CAD model; we generated it using CloudCompare)

tac1.pth (a combination of the two files above, used to train OverlapPredator)
