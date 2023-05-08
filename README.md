PBSHM_mdof
==============================

In this project, I try to simulate an Mdof sys all of this is done in the context of population based Structural Health Monitroing, 
The folder PBSHM contains the main code and notebooks folder is used to generate figures.
If you want to replicate the result, you have to first load the project and install the packages PBSHM_mdof 
The folder PBSHM_mdof/system contains the tools for simulating a population of mdof system (time domain simulation) Including the effect of envirement on the structure adding anomaly .
The folder PBSHM_mdof/data : contains a file create_dataset, where we generate the time_domain data of the 20 8-dof system and save them in an HDF5 file format. the structure of the HDF5 file is as follow:

-  The top-level group is a population group, with the name of the population being the name of the group. Each population group has a group named "population_params" that contains the parameters for the systems in the population.
- Each system in the population has a group named after the system name under the "population_params" group. Each system group contains datasets for the mass, stiffness, and damping parameters.
Under each population group, there can be multiple simulation groups, named after the simulation. Each simulation group contains attributes for the simulation parameters, such as time step (dt), end time (t_end), and standard deviation of the latent variables (std_latent).
- Each simulation group contains multiple experiment groups, named after the experiment. Each experiment group contains attributes for the experiment parameters, such as the latent value, anomaly level, state, input location, and amplitude.
-Each experiment group contains a group named "resonance_frequency" that contains datasets for the resonance frequencies of the systems in the population, and a group named "stiffness" that contains datasets for the stiffness parameters of the systems in the population.
- Each experiment group also contains a group named "TDD" that contains datasets for time-domain data for the experiment, 
THe data in a tree format is as follow
Population Group
    "population_params" Group
        System Group
            Mass Dataset
            Stiffness Dataset
            Damping Dataset
    Simulation Group
        Resonance Frequency Group
            Frequency Datasets for each system in the population
        Stiffness Group
            Stiffness Datasets for each system in the population
        Experiment Group
            Latent Value Attribute
            Anomaly Level Attribute
            State Attribute
            Input Location Attribute
            Amplitude Attribute    
        Time-Domain Data Group (TDD)
            Output Dataset

As can be seen all the meta data is saved for each similation use the .visit(print) command to explore the file.
 
The file process_data, load the TDD and compute the PSD with different noise level and save the result in parquet file 
==> the data can be found on Zenodo.

The folder PBSHM_mdof/models handels the model implimentaion (baseline and the zero-shot classifier for anomaly detection). on the baseline subfolder the baseline method is defined. In train_model.py we train the classifier for zero-shot anomaly detection. The current implimentation allows for saving the result in MLFLOW

docs folder contains images used for publications.
Only the processed data is available in zenodo.
