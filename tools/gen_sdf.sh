urdf_folder="./src/tvmd/urdf"
urdf_fname="${urdf_folder}/model.urdf"

xacro ${urdf_folder}/tvmd.xacro > $urdf_fname
gz sdf -p $urdf_fname > ${urdf_folder}/model.sdf
