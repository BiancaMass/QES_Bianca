## DSDI set up
If issue installing venv do the following:

- `apt-get update`
- `apt-get install python3.10-venv`

Then, create the venv: 
`python3.10 -m venv venv1`

And install requirements:
`pip install -r requirements.txt `

You might have to change the root path to the right one from the terminal (in case it gives an 
error of finding the right modules):

`export PYTHONPATH=$PYTHONPATH:/root/QES_Bianca
`

### Credits
This folder contains the project of the Quantum Evolutionary Strategy for function optimization 
(Vincenzo Lipardi) adapted for a image generation GAN (PQWGAN) as part of Bianca Massacci's 
master thesis project for the master in data science for decision-making at Maastricht University.

TODO: describe complete dir structure, cite sources, explain how to run it, explain the 
algorithm, give credits where needed, show example results, describe what each script does.

