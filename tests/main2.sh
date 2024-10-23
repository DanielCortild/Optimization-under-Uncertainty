#!/bin/sh
#SBATCH -J SP2                                  # Job name
#SBATCH -N 1                                    # Nodes requested
#SBATCH -n 1                                    # Tasks requested
#SBATCH --exclusive                             # Exclusivity requested
#SBATCH -t 12:00:00                             # Time requested in hour:minute:second
#SBATCH --output=output/output/output_%j.txt    # Output file
#SBATCH --error=output/error/error_%j.txt       # Error file

cd ..
. venv/bin/activate
cd tests
module load Python/3.11.5-GCCcore-13.2.0
python3 main2.py $*
