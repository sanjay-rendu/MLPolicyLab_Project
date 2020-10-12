# Instructions to run an experiment

## 1. Setup

#### 1.1 Install daggit library

Binaries for this library can be found [here](https://github.com/sanjay-rendu/sunbird-ml-workbench/tree/master/bin) \
`pip install daggit-0.5.0.tar.gz`

 Note: Nothing to do here if you're using `bills3` virtualenv

#### 1.2 Install other dependencies

List of other requirements for this project can be found in requirements.txt in the root directory of this project\
 `pip install -r requirements.txt` 
 
  Note: Nothing to do here if you're using `bills3` virtualenv
 
 #### 1.3 Set up environment variables in .bashrc
 `export DAGGIT_HOME=/PATH/TO/FOLDER` (intermediate results of the pipeline will be saved here) \
 `export daggit_extra=/PATH/TO/PYTHON/MODULE` (directory location of `pipeline` module that contains node definitions for this project)
 
 Note: Current settings to add to `.bashrc` in your user home directory\
 `export DAGGIT_HOME="/data/groups/bills3/vrenduch/DAGGIT_HOME"`\
`export daggit_extra="/data/groups/bills3/mlpp_project_home/pipeline"`
 
 #### 1.4 Define Postgres Hook to connect to the project database
 We use airflow's Postgres Hook to manage connection to the project database. More details on managing the connection can be found [here](https://airflow.apache.org/docs/stable/howto/connection/index.html). 
 
 Note: Nothing to do here if you're using `bills3` virtualenv
 
 ## 2. Running the project pipeline
 
 #### 2.1 Register an experiment definition by its name
 `daggit init /PATH/TO/EXPERIMENT.yaml`
 
 Note: This is a one time process to register an experiment config file (yaml). Future edits to the registered yaml file will be automatically be reflected in the DAG. 
 
 #### 2.2 Run the experiment
 To run the experiment: `daggit run EXPERIMENT_NAME`  
 To re-run only the nodes that previously failed: `daggit run --clear_failed EXPERIMENT_NAME` \
 To re-run all the nodes: `daggit run --clear_all EXPERIMENT_NAME` 
 
 Note: EXPERIMENT_NAME is found on the first line of the experiment config (yaml)
 
 #### [Optional] 2.3 Monitoring DAG and node run status
 We can use [airflow's UI](https://airflow.apache.org/docs/stable/ui.html) to monitor if the nodes have successfully completed their tasks and use task logs to debug the nodes \
 `airflow webserver -p 8080` (starts airflow webserver on port 8080)
 
