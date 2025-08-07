#!/usr/bin/env bash

set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}[1/6] Ensuring ~/.local/bin exists...${NC}"
mkdir -p "$HOME/.local/bin"

echo -e "${GREEN}[2/6] Checking for Java...${NC}"
if ! java -version &>/dev/null; then
  echo -e "${RED}Java is not properly installed or configured.${NC}"
  echo -e "Please install Java using one of the following options:"
  echo -e " • ${GREEN}sdk install java 17.0.8-tem${NC} (via https://sdkman.io)"
  echo -e " • ${GREEN}brew install openjdk@17${NC} (macOS)"
  echo -e " • ${GREEN}apt install openjdk-17-jdk${NC} (Linux)"
  echo -e " • ${GREEN}you might have to add the java path to your PATH variable${NC}"
  exit 1
fi

echo -e "${GREEN}[3/6] Checking for conda...${NC}"
# Check for conda in multiple possible locations
CONDA_FOUND=false
if command -v conda &>/dev/null; then
  CONDA_FOUND=true
elif [[ -f "$HOME/miniconda3/bin/conda" ]]; then
  export PATH="$HOME/miniconda3/bin:$PATH"
  CONDA_FOUND=true
elif [[ -f "$HOME/anaconda3/bin/conda" ]]; then
  export PATH="$HOME/anaconda3/bin:$PATH"
  CONDA_FOUND=true
elif [[ -f "/opt/conda/bin/conda" ]]; then
  export PATH="/opt/conda/bin:$PATH"
  CONDA_FOUND=true
fi

if [[ "$CONDA_FOUND" == "true" ]]; then
  echo -e "${GREEN}✅ Conda found: $(conda --version)${NC}"
else
  echo -e "${YELLOW}Conda not found. Installing Miniconda...${NC}"
  
  # Detect OS and architecture
  OS=$(uname -s | tr '[:upper:]' '[:lower:]')
  ARCH=$(uname -m)
  
  if [[ "$ARCH" == "x86_64" ]]; then
    ARCH="x86_64"
  elif [[ "$ARCH" == "arm64" ]] || [[ "$ARCH" == "aarch64" ]]; then
    # For macOS, use arm64; for Linux, use aarch64
    if [[ "$OS" == "darwin" ]]; then
      ARCH="arm64"
    else
      ARCH="aarch64"
    fi
  else
    echo -e "${RED}Unsupported architecture: $ARCH${NC}"
    exit 1
  fi
  
  # Use the correct Miniconda URL format
  if [[ "$OS" == "darwin" ]]; then
    # macOS
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-${ARCH}.sh"
  elif [[ "$OS" == "linux" ]]; then
    # Linux
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${ARCH}.sh"
  else
    echo -e "${RED}Unsupported operating system: $OS${NC}"
    exit 1
  fi
  
  echo -e "${GREEN}Downloading Miniconda from: $MINICONDA_URL${NC}"
  
  # Download with better error handling
  if ! curl -L "$MINICONDA_URL" -o miniconda.sh; then
    echo -e "${RED}Failed to download Miniconda. Please check your internet connection.${NC}"
    exit 1
  fi
  
  # Verify the download is a valid installer (check if it starts with shebang or is executable)
  if [[ ! -f miniconda.sh ]] || [[ ! -s miniconda.sh ]]; then
    echo -e "${RED}Download failed - file is empty or missing.${NC}"
    echo -e "${YELLOW}Please try installing conda manually:${NC}"
    echo -e "${GREEN}  brew install --cask miniconda${NC} (macOS with Homebrew)"
    echo -e "${GREEN}  Or visit: https://docs.conda.io/en/latest/miniconda.html${NC}"
    rm -f miniconda.sh
    exit 1
  fi
  
  # Check if file is executable or has proper installer signature
  if ! head -1 miniconda.sh | grep -q "^#!" && ! file miniconda.sh | grep -q "executable"; then
    echo -e "${RED}Download failed - file is not a valid installer.${NC}"
    echo -e "${YELLOW}Please try installing conda manually:${NC}"
    echo -e "${GREEN}  brew install --cask miniconda${NC} (macOS with Homebrew)"
    echo -e "${GREEN}  Or visit: https://docs.conda.io/en/latest/miniconda.html${NC}"
    rm -f miniconda.sh
    exit 1
  fi
  
  # Check if miniconda3 directory already exists
  if [[ -d "$HOME/miniconda3" ]]; then
    echo -e "${YELLOW}Miniconda directory already exists at $HOME/miniconda3${NC}"
    echo -e "${YELLOW}Adding to PATH for current session...${NC}"
    export PATH="$HOME/miniconda3/bin:$PATH"
  else
    # Install Miniconda
    bash miniconda.sh -b -p "$HOME/miniconda3"
    rm miniconda.sh
    
    # Add conda to PATH for current session
    export PATH="$HOME/miniconda3/bin:$PATH"
    
    # Initialize conda for the current shell
    source "$HOME/miniconda3/bin/activate"
    conda init bash
    if [[ -n "${ZSH_VERSION:-}" ]]; then
      conda init zsh
    fi
    
    echo -e "${YELLOW}⚠️  Conda has been installed but needs shell initialization.${NC}"
    echo -e "${YELLOW}Please run one of the following commands to activate conda:${NC}"
    echo -e "${GREEN}  source ~/.bashrc${NC} (for bash)"
    echo -e "${GREEN}  source ~/.zshrc${NC} (for zsh)"
    echo -e "${YELLOW}Or restart your terminal session.${NC}"
    echo -e "${YELLOW}After that, you may need to run this setup script again.${NC}"
  fi
  
  # Ensure conda is available for the rest of the script
  export PATH="$HOME/miniconda3/bin:$PATH"
  
  # Initialize conda for the current session if not already done
  if ! command -v conda &>/dev/null; then
    echo -e "${YELLOW}Initializing conda for current session...${NC}"
    export PATH="$HOME/miniconda3/bin:$PATH"
  fi
fi

echo -e "${GREEN}[4/6] Checking for mamba...${NC}"
if ! command -v mamba &>/dev/null; then
  echo -e "${YELLOW}Mamba not found. Installing mamba...${NC}"
  
  # Ensure conda is available
  if ! command -v conda &>/dev/null; then
    echo -e "${RED}Conda is not available. Please restart your terminal and run this script again.${NC}"
    exit 1
  fi
  
  # Accept Terms of Service for required channels
  #echo -e "${GREEN}Accepting conda channel Terms of Service...${NC}"
  #conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
  #conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
  #conda tos accept --override-channels --channel https://conda-forge.org/channels/conda-forge
  
  # Install mamba using conda
  conda install mamba -n base -c conda-forge -y
  
  echo -e "${GREEN}✅ Mamba installed successfully${NC}"
else
  echo -e "${GREEN}✅ Mamba found: $(mamba --version | head -n1)${NC}"
fi

echo -e "${GREEN}[5/6] Downloading Nextflow...${NC}"
curl -s https://get.nextflow.io | bash
chmod +x nextflow
mv nextflow "$HOME/.local/bin/"

echo -e "${GREEN}[6/6] Ensuring ~/.local/bin is in PATH...${NC}"

# Determine the correct shell profile file
determine_profile_file() {
  if [[ -n "${ZSH_VERSION:-}" ]]; then
    echo "${HOME}/.zshrc"
  elif [[ -n "${BASH_VERSION:-}" ]]; then
    echo "${HOME}/.bashrc"
  elif [[ -f "${HOME}/.profile" ]]; then
    echo "${HOME}/.profile"
  else
    # Fallback based on current shell
    case "$SHELL" in
      */zsh) echo "${HOME}/.zshrc" ;;
      */bash) echo "${HOME}/.bashrc" ;;
      *) echo "${HOME}/.profile" ;;
    esac
  fi
}

PROFILE_FILE=$(determine_profile_file)
echo -e "${GREEN}Using profile file: $PROFILE_FILE${NC}"

if ! echo "$PATH" | grep -q "$HOME/.local/bin"; then
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$PROFILE_FILE"
  echo -e "${GREEN}Updated $PROFILE_FILE to include ~/.local/bin in PATH.${NC}"
fi

echo -e "${GREEN}✅ Setup complete!${NC}"
echo -e "${GREEN}Run 'nextflow -version' to verify Nextflow installation.${NC}"
echo -e "${GREEN}Run 'conda --version' and 'mamba --version' to verify conda/mamba installation.${NC}"

# Final verification and initialization
echo -e "${GREEN}[Final] Ensuring conda and mamba are available...${NC}"

# Ensure conda is in PATH for the current session
export PATH="$HOME/miniconda3/bin:$PATH"

# Test if conda and mamba are working
if command -v conda &>/dev/null && command -v mamba &>/dev/null && command -v nextflow &>/dev/null; then
  echo -e "${GREEN}🎉 All tools are ready to use!${NC}"
  echo -e "${GREEN}✅ Conda version: $(conda --version)${NC}"
  echo -e "${GREEN}✅ Mamba version: $(mamba --version | head -n1)${NC}"
  echo -e "${GREEN}✅ Nextflow version: $(nextflow -version | head -n1)${NC}"
else
  echo -e "${YELLOW}⚠️  Some tools may need shell restart to be available.${NC}"
  echo -e "${YELLOW}Please restart your terminal or run: source $PROFILE_FILE${NC}"
  echo -e "${YELLOW}Alternatively, you can manually add to PATH: export PATH=\"\$HOME/miniconda3/bin:\$PATH\"${NC}"
fi

# Provide instructions for permanent setup
echo -e "${GREEN}📝 For permanent access, add this line to your shell profile ($PROFILE_FILE):${NC}"
echo -e "${GREEN}   export PATH=\"\$HOME/miniconda3/bin:\$PATH\"${NC}"

# Automatically add conda to the appropriate shell profile
if ! grep -q "miniconda3/bin" "$PROFILE_FILE"; then
  echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> "$PROFILE_FILE"
  echo -e "${GREEN}✅ Added conda to $PROFILE_FILE for permanent access${NC}"
else
  echo -e "${GREEN}✅ Conda PATH already exists in $PROFILE_FILE${NC}"
fi

echo -e "${GREEN}📝 To activate in current session, run: source $PROFILE_FILE${NC}"
echo -e "${GREEN}📝 For new terminal sessions, conda and mamba will be automatically available${NC}"