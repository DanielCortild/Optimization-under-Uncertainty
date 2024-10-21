#!/bin/sh
#SBATCH -J stopro                               # Job name
#SBATCH -N 4                                    # Nodes requested
#SBATCH -n 1                                    # Tasks requested
#SBATCH --exclusive                             # Exclusivity requested
#SBATCH -t 6:00:00                              # Time requested in hour:minute:second
#SBATCH --output=output/output/output_%j.txt    # Output file
#SBATCH --error=output/error/error_%j.txt       # Error file

. venv/bin/activate
module load Python/3.11.5-GCCcore-13.2.0
python3 L2.py $*
