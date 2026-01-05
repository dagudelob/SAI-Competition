# SAI-Competition 

## install SAI competition APP
https://docs.competesai.com/getting-started/installation 

Installation
Add our Homebrew Tap to your device:


 
brew tap arenax-labs/tap  
### /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" 
    echo >> /home/dagudelo/.zshrc
    echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"' >> /home/dagudelo/.zshrc
    eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"
   
         echo >> /home/dagudelo/.zshrc
         sudo apt-get install build-essential
   
    Install SAI:
        brew update

    INSTALL SAI    
        brew install sai
        brew install cbonsai
    
# fOR Python
pip install sai-rl

## install requeriments.txt
pip install -r requeriments.txt

### freeze > Requirements.txt
pip install sai-mujoco
pip install sai-pygame