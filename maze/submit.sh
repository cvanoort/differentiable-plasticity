# This job needs 1 compute node with 2 processors per node.
#PBS -l nodes=1:ppn=2,pmem=4gb,pvmem=5gb
# Use the long queue to allow for longer runtime
#PBS -q l1wkq
# It should be allowed to run for up to 1 hour.
#PBS -l walltime=04:00:00:00
# Name of job.
#PBS -N plasticity_maze
# Join STDERR TO STDOUT.  (omit this if you want separate STDOUT AND STDERR)
#PBS -j oe
SEED=$(date +"%s")
python maze.py --type plastic --rng_seed $SEED
