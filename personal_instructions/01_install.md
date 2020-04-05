# How to install my local environment for the Udacity Course Deep Reinforcement Learning - starting from April 2020

## Editor
1. Download Visual Studio Code
https://code.visualstudio.com/docs/?dv=linux64_deb
2. Install 
sudo dpkg -i code_1.43.2-1585036376_amd64.deb

## Anaconda3 Python Environment
following the procedure https://docs.anaconda.com/anaconda/install/linux/
1. Download Anaconda Installer for python3.7
https://www.anaconda.com/distribution/#linux
2. Open new terminal
`<ctrl>+<alt>t`


3. Install dependencies
```bash
sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
```
3. Local Install of anaconda in home folder
```bash
bash ~/Downloads/Anaconda3-2020.02-Linux-x86_64.sh
```
4. Create environment for udacity course
```bash
conda create -n udacity python=3.7
```
5. Start environment automatically than open a terminal
```bash
echo "conda activate udacity">>~/.bashrc
```
6. Close current terminal
```bash
exit
```
7. Open new terminal
`<ctrl>+<alt>t`
8. Install gym and dependencies
following the instructions from https://github.com/openai/gym
I added gym as submodule to the fork of the udacity repository by
```bash
git submodule add https://github.com/openai/gym gym
```
  * Base install, ignore the errors because of mujoco-py
```bash
cd gym
pip install -e .
pip install -e '.[all]'
```
  * Install jupyter framework and dependencies seaborn
```bash
conda install jupyter matplotlib
```
9 Start jupyter without password protection for local use
```bash
cd <deep-reinforcement-learning>
jupyter notebook --ip='*' --NotebookApp.token='' --NotebookApp.password=''
```