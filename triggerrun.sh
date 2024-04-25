#!/bin/bash

echo "Press 'Enter' to launch dualboot"
echo "Press 'x' to exit the launcher"

while true; do
    read -rsn1 key

    if [[ $key == "" ]]; then
        echo "Launching dualboot..."
        python3 DualBootV4.1.py
        echo "dualboot has exited"
    elif [[ $key == "x" ]]; then
        echo "Exiting the launcher..."
        break
    fi
done