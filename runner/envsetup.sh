#!/bin/bash

# setting bash source.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: This script should be sourced into the current shell.  Use the following syntax:"
    echo ""
    echo "    source scripts/envsetup.sh"
    echo ""
    exit 1
fi

export REPO_TOP="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

function tao_ws {
   python $REPO_TOP/runner/tao_ws.py "$@"
}
export -f tao_ws

# Checks if 'i' is included in $-, if yes -> interactive
function _check_shell_is_interactive() {
    case "${-}" in
        *i*)    return 0 ;;
          *)    return 1 ;;
    esac
}

# Print the build message.
function help() {
    if _check_shell_is_interactive ; then
        cat <<EOF

Build environment set up.

The following environment variables have been set:

  REPO_TOP       $REPO_TOP

The following functions have been added to your environment:

  tao_ws                 Run command inside the container.
EOF
    fi
}

function _check_requirements(){
    warnings=()

    # Check python.
    if ! command -v python >/dev/null; then
        echo -e "\033[1;31mERROR:\033[0m python not found"
        return 1
    fi

    # Check if docker was installed.
    if ! command -v docker >/dev/null; then
        warnings+=("docker not found")
    else
        if ! id -nG | grep -qw "docker"; then
            [[ $OSTYPE = darwin* ]] || warnings+=("You should add yourself to the docker group by running \"sudo usermod -a -G docker $(whoami)\"")
        fi
        if ! grep -q "nvcr.io" $HOME/.docker/config.json; then
            warnings+=("You should login to container registry by running \"docker login nvcr.io\"")
        fi
    fi

    help

    for w in "${warnings[@]}"; do
        echo -e "\033[1;33mWARNING:\033[0m $w"
    done
}

_check_requirements || echo -e "\033[1;31mBuild environment setup failed.\033[0m"
