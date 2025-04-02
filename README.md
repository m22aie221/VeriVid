cd /scratch/data/m22aie221/workspace/VeriVid
python /scratch/data/m22aie221/workspace/VeriVid/parse_dataset_config.py

python select_train_test_candidate.py

sbatch run_VeriVid_extract_frames_1.sh

/scratch/data/m22aie221/workspace/SEA-RAFT/SEA-RAFT/run_extract_optical_2.sh
cd /scratch/data/m22aie221/workspace/SEA-RAFT/SEA-RAFT
conda activate SEA-RAFT
sbatch run_extract_optical_2.sh

cd /scratch/data/m22aie221/workspace/midas/MiDaS
conda activate midas-py310
sbatch run_midas.sh


cd /scratch/data/m22aie221/workspace/dinov2/dinov2
conda activate DINO
sbatch run_dino_features.sh


cd /scratch/data/m22aie221/workspace/VeriVid
conda activate DINO
 
cd /scratch/data/m22aie221/workspace/VeriVid/TimeSformer
conda activate DINO_clone

cd /scratch/data/m22aie221/workspace/VeriVid
conda activate DINO
sbatch run_incremental_train_transfromer.sh