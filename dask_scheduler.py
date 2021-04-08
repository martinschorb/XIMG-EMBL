
import sys, time

import dask
import dask.array as da


from dask_jobqueue import SLURMCluster
import os


#-----------------------------------------


#         DASK 

# ----------------------------------------




worker_cpu = 2

workers_per_job = 12


cluster = SLURMCluster(
    cores=worker_cpu * workers_per_job,
    processes=workers_per_job,
    memory="32GB",
    shebang='#!/usr/bin/env bash',
    walltime="01:30:00",
    local_directory='/tmp',
    death_timeout="15s")


maxnodes = 30

ca = cluster.adapt(
    minimum = workers_per_job, maximum=maxnodes * workers_per_job,
    #target_duration="360s",  # measured in CPU time per worker
                             # -> 30 seconds at 12 cores / worker
    # scale_factor=1.0  # prevent from scaling up because of CPU or MEM need
)


print('Dask cluster scheduler running at '+cluster.scheduler_address+'\n\n\n')

sec = input('Type any input to stop the scheduler.\n')

