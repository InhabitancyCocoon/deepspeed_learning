#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0



deepspeed \
   --num_nodes=1 \
   --num_gpus=4 \
   --master_port=23142 \
   pretrain.py
