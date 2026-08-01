#!/bin/bash

ls -l raw_data/full_dataset/ | grep -i "augm"

rm raw_data/full_dataset/*augm*

ls -l raw_data/full_dataset/ | grep -i "augm"
